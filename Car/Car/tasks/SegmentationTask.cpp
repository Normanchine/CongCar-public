#include "SegmentationTask.h"
#include "../common/PerfStats.h"
#include "../common/SharedResult.h"
#include "../inference/RKNNEngine.h"

#include "../thread/ThreadTaskFactory.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

REGISTER_THREAD_TASK(SegmentationTask)

SegmentationTask::SegmentationTask(SharedContext* ctx) : IThreadTask(ctx) {}
SegmentationTask::~SegmentationTask() = default;

void SegmentationTask::run() {
    RKNNEngine engine;
    const auto& ai = ctx_->config.ai;

    if (!engine.load(ai.segmentation_model_path, 2)) {
        std::cerr << "[SegTask] 模型加载失败" << std::endl;
        return;
    }

    std::cout << "[SegTask] 已启动" << std::endl;

    const int seg_w = engine.inputWidth();
    const int seg_h = engine.inputHeight();
    const int seg_n = seg_w * seg_h;

    // 首帧预提交
    cv::Mat input;
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        if (ctx_->segmentation_queue.tryConsume(input)) break;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    if (ctx_->stop.load(std::memory_order_acquire)) return;

    auto future = engine.submit(std::move(input));

    uint64_t frame_counter = 0;

    while (!ctx_->stop.load(std::memory_order_acquire)) {

        cv::Mat next_input;
        bool has_frame = false;
        while (!ctx_->stop.load(std::memory_order_acquire)) {
            if (ctx_->segmentation_queue.tryConsume(next_input)) {
                has_frame = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        if (!has_frame) break;

        auto t1 = std::chrono::steady_clock::now();

        auto outputs = future.get();

        auto t2 = std::chrono::steady_clock::now();
        future = engine.submit(std::move(next_input));

        if (!outputs.empty()) {
            int orig_w = ctx_->frame_width.load(std::memory_order_acquire);
            int orig_h = ctx_->frame_height.load(std::memory_order_acquire);

            auto tp0 = std::chrono::steady_clock::now();

            // argmax: output shape (2, seg_n) CV_32FC1 — ch0=背景, ch1=赛道
            const cv::Mat& out = outputs[0];
            const float* ch0 = out.ptr<float>(0);
            const float* ch1 = out.ptr<float>(1);

            cv::Mat mask(seg_h, seg_w, CV_8UC1);
            for (int i = 0; i < seg_n; i++) {
                mask.data[i] = (ch0[i] >= ch1[i]) ? 0 : 1;
            }

            auto tp1 = std::chrono::steady_clock::now();

            cv::Mat mask_resized;
            cv::resize(mask, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

            auto tp2 = std::chrono::steady_clock::now();

            SegResult result;
            result.mask      = std::move(mask_resized);
            result.width     = orig_w;
            result.height    = orig_h;
            result.frame_id  = ++frame_counter;
            ctx_->seg_result.publish(std::move(result));
            ctx_->seg_cv_.notify_one();  // 唤醒 ControlTask

            auto tp3 = std::chrono::steady_clock::now();
            auto argmax_us = std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count();
            auto resize_us = std::chrono::duration_cast<std::chrono::microseconds>(tp2 - tp1).count();
            auto pub_us    = std::chrono::duration_cast<std::chrono::microseconds>(tp3 - tp2).count();
            std::cout << "[SegPost] argmax=" << argmax_us << "us  resize=" << resize_us
                      << "us  publish=" << pub_us << std::endl;
        }

        auto t3 = std::chrono::steady_clock::now();
        auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto post_us  = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        ctx_->perf_stats.recordStages(SLOT_SEGMENTATION, 0, infer_us, post_us);
    }

    std::cout << "[SegTask] 已退出" << std::endl;
}
