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

    // 双引擎：Core1 + Core2 交替推理
    RKNNEngine eng_a, eng_b;
    if (!eng_a.load(seg_cfg.model_path, 1)) {
        std::cerr << "[SegTask] eng_a(Core1) 加载失败" << std::endl;
        return;
    }
    if (!eng_b.load(seg_cfg.model_path, 2)) {
        std::cerr << "[SegTask] eng_b(Core2) 加载失败" << std::endl;
        return;
    }

    const int seg_w = eng_a.inputWidth();
    const int seg_h = eng_a.inputHeight();
    const int seg_n = seg_w * seg_h;

    std::cout << "[SegTask] 双核已启动 (Core1+Core2)" << std::endl;

    // 首两帧预提交（A先, B后）
    cv::Mat input;
    auto consume = [&]() -> bool {
        while (!ctx_->stop.load(std::memory_order_acquire))
            if (ctx_->segmentation_queue.tryConsume(input)) return true;
            else std::this_thread::sleep_for(std::chrono::microseconds(500));
        return false;
    };

    if (!consume()) return;
    auto fut_a = eng_a.submit(std::move(input));
    if (!consume()) return;
    auto fut_b = eng_b.submit(std::move(input));

    uint64_t frame_counter = 0;
    bool toggle = false;  // false→get A, true→get B

    while (!ctx_->stop.load(std::memory_order_acquire)) {

        if (!consume()) break;

        auto t_start = std::chrono::steady_clock::now();

        // 交替：偶数帧取A结果+提交A，奇数帧取B结果+提交B
        auto& eng    = toggle ? eng_b    : eng_a;
        auto& fut    = toggle ? fut_b    : fut_a;
        toggle = !toggle;

        auto [outputs, infer_us] = fut.get();
        fut = eng.submit(std::move(input));

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

    std::cout << "[SegTask] 双核已退出" << std::endl;
}
