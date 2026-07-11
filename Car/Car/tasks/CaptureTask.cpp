#include "CaptureTask.h"
#include "../common/PerfStats.h"
#include "../hal/memory/shared_memory_manager.h"
#include "../thread/ThreadTaskFactory.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

REGISTER_THREAD_TASK(CaptureTask)

CaptureTask::CaptureTask(SharedContext* ctx) : IThreadTask(ctx) {}
CaptureTask::~CaptureTask() = default;

void CaptureTask::run() {
    auto shm = hal::memory::SharedMemoryManager::create();

    const auto& shm_name = ctx_->config.shm_name;
    const auto  shm_size = ctx_->config.shm_size;
    const int   det_w  = ctx_->config.ai.detection.input_w;
    const int   det_h  = ctx_->config.ai.detection.input_h;
    const int   seg_w  = ctx_->config.ai.segmentation.input_w;
    const int   seg_h  = ctx_->config.ai.segmentation.input_h;

    uint64_t last_fid = 0;

    while (!ctx_->stop.load(std::memory_order_acquire)) {

        // ---- 1. 连接共享内存（支持断线重连） ----
        while (!shm->isValid() && !ctx_->stop.load(std::memory_order_acquire)) {
            if (shm->open(shm_name, shm_size)) {
                std::cout << "[CaptureTask] 已连接到共享内存: " << shm_name << std::endl;
                last_fid = 0;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (ctx_->stop.load(std::memory_order_acquire)) break;

        auto t_start = std::chrono::steady_clock::now();

        // ---- 2. 读取header: fid(0) | width(8) | height(12) ----
        auto* header = static_cast<uint8_t*>(shm->getBuffer());
        if (!header) {
            shm->close();
            continue;
        }

        uint64_t fid;
        uint32_t w, h;
        std::memcpy(&fid, header,      sizeof(uint64_t));
        std::memcpy(&w,   header + 8,  sizeof(uint32_t));
        std::memcpy(&h,   header + 12, sizeof(uint32_t));

        // 帧号未变 → 根据当前采集帧率动态调节等待时间
        if (fid == last_fid) {
            double fps = ctx_->perf_stats.getFps(SLOT_CAPTURE);
            int sleep_us = 2000;
            if (fps > 0.0) {
                int interval_us = static_cast<int>(1'000'000.0 / fps);
                sleep_us = std::clamp(interval_us / 4, 100, 2000);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            continue;
        }

        // 无效尺寸
        if (w == 0 || h == 0 || w > 4096 || h > 4096) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        // ---- 3. 深拷贝图像数据 ----
        size_t data_size = static_cast<size_t>(w) * h * 3;
        if (hal::memory::SHM_HEADER_SIZE + data_size > shm_size) {
            std::cerr << "[CaptureTask] 图像数据超出共享内存大小" << std::endl;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        void* shm_data = shm->getDataBuffer();
        cv::Mat frame(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
        std::memcpy(frame.data, shm_data, data_size);

        last_fid = fid;

        // ---- 4. 垂直翻转（Python 端做了 flip(0)，这里还原） ----
        cv::flip(frame, frame, 0);
        // 注：frame保持RGB格式（共享内存原始格式），模型直接用，显示时再转BGR

        // ---- 5. 记录全图尺寸（供DetTask坐标映射，检测走全图不做裁剪） ----
        ctx_->full_frame_width.store(static_cast<int>(w), std::memory_order_release);
        ctx_->full_frame_height.store(static_cast<int>(h), std::memory_order_release);

        // ---- 6. 检测帧：全图 resize 入队（检测需要看到完整画面） ----
        cv::Mat det_input;
        cv::resize(frame, det_input, cv::Size(det_w, det_h));
        ctx_->detection_queue.produce(det_input);

        // ---- 7. 显示帧 + 工作帧：全图（crop 之前） ----
        ctx_->work_frame_queue.produce(frame.clone());  // TrafficSignScene 用全图裁剪标志
        if (ctx_->config.debug_mode) {
            ctx_->display_queue.produce(frame.clone()); // 显示全图
        }

        // ---- 8. 帧裁剪（仅影响分割和巡线） ----
        if (ctx_->config.frame_crop.enable) {
            cv::Rect crop(ctx_->config.frame_crop.x, ctx_->config.frame_crop.y,
                          ctx_->config.frame_crop.w, ctx_->config.frame_crop.h);
            frame = frame(crop).clone();  // clone确保内存连续
            w = static_cast<uint32_t>(ctx_->config.frame_crop.w);
            h = static_cast<uint32_t>(ctx_->config.frame_crop.h);
        }

        // ---- 9. 记录工作帧尺寸（供SegTask坐标映射和鸟瞰Mask还原） ----
        ctx_->frame_width.store(static_cast<int>(w), std::memory_order_release);
        ctx_->frame_height.store(static_cast<int>(h), std::memory_order_release);

        // ---- 10. 分割帧：裁剪后 resize 入队 ----
        cv::Mat seg_input;
        cv::resize(frame, seg_input, cv::Size(seg_w, seg_h));
        ctx_->segmentation_queue.produce(seg_input);

        // ---- 11. 记录本帧耗时 ----
        auto t_end = std::chrono::steady_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();
        ctx_->perf_stats.record(SLOT_CAPTURE, elapsed_us);
    }

    shm->close();
    std::cout << "[CaptureTask] 已退出" << std::endl;
}
