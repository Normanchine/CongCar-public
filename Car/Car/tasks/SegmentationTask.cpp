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
    const auto& seg_cfg = ctx_->config.ai.segmentation;

    // 单引擎：Core2 推理
    RKNNEngine engine;
    if (!engine.load(seg_cfg.model_path, 2)) {
        std::cerr << "[SegTask] Core2 加载失败" << std::endl;
        return;
    }

    const int seg_w = engine.inputWidth();
    const int seg_h = engine.inputHeight();
    const int seg_n = seg_w * seg_h;

    std::cout << "[SegTask] 单核已启动 (Core2)" << std::endl;

    // 流水线首帧预提交
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

        auto t_start = std::chrono::steady_clock::now();

        auto [outputs, infer_us] = future.get();

        future = engine.submit(std::move(next_input));

        auto t_post_start = std::chrono::steady_clock::now();

        if (!outputs.empty()) {
            int orig_w = ctx_->frame_width.load(std::memory_order_acquire);
            int orig_h = ctx_->frame_height.load(std::memory_order_acquire);

            const cv::Mat& out = outputs[0];
            const float* ch0 = out.ptr<float>(0);
            const float* ch1 = out.ptr<float>(1);

            cv::Mat mask(seg_h, seg_w, CV_8UC1);
            for (int i = 0; i < seg_n; i++) {
                mask.data[i] = (ch0[i] >= ch1[i]) ? 0 : 1;
            }

            // ---- 去噪：连通域过滤，长宽任一不足 min_region_px 的碎片剪掉 ----
            // 赛道干扰物被分割成小块会污染边界提取；在源头(模型输出mask)清除，
            // 所有下游（巡线 + 显示）拿到的都是干净 mask
            if (seg_cfg.min_region_px > 0) {
                cv::Mat labels, stats, centroids;
                int ncomp = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
                for (int i = 1; i < ncomp; ++i) {   // 0=背景
                    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
                    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                    if (w < seg_cfg.min_region_px || h < seg_cfg.min_region_px)
                        mask.setTo(0, labels == i);   // 长宽任一不足 → 剪掉
                }
            }

            cv::Mat mask_resized;
            cv::resize(mask, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

            SegResult result;
            result.mask      = std::move(mask_resized);
            result.width     = orig_w;
            result.height    = orig_h;
            result.frame_id  = ++frame_counter;
            ctx_->seg_result.publish(std::move(result));
            ctx_->seg_cv_.notify_one();
        }

        auto t_end = std::chrono::steady_clock::now();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        auto post_us  = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_post_start).count();
        ctx_->perf_stats.recordStages(SLOT_SEGMENTATION, total_us, infer_us, post_us);
    }

    std::cout << "[SegTask] 单核已退出" << std::endl;
}
