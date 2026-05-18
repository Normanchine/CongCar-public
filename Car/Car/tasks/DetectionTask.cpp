#include "DetectionTask.h"
#include "../common/PerfStats.h"
#include "../common/SharedResult.h"
#include "../inference/RKNNEngine.h"
#include "../inference/YoloPostProcess.h"

#include "../thread/ThreadTaskFactory.h"
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

REGISTER_THREAD_TASK(DetectionTask)

DetectionTask::DetectionTask(SharedContext* ctx) : IThreadTask(ctx) {}
DetectionTask::~DetectionTask() = default;

void DetectionTask::run() {
    RKNNEngine engine;
    yolo::PostProcessBuffers ppbuf;
    const auto& ai = ctx_->config.ai;

    if (!engine.load(ai.detection_model_path, 0)) {
        std::cerr << "[DetTask] 模型加载失败" << std::endl;
        return;
    }

    const int iw = ai.detection_input_w;
    const int ih = ai.detection_input_h;

    std::cout << "[DetTask] 已启动 (输入: " << iw << "x" << ih << ")" << std::endl;

    // 流水线首帧预提交（CaptureTask 已完成 BGR→RGB + resize）
    cv::Mat input;
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        if (ctx_->detection_queue.tryConsume(input)) break;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    if (ctx_->stop.load(std::memory_order_acquire)) return;

    auto future = engine.submit(std::move(input));

    uint64_t frame_counter = 0;

    while (!ctx_->stop.load(std::memory_order_acquire)) {

        cv::Mat next_input;
        bool has_frame = false;
        while (!ctx_->stop.load(std::memory_order_acquire)) {
            if (ctx_->detection_queue.tryConsume(next_input)) {
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

            std::vector<yolo::RawBox> all_raw;
            int n_out = static_cast<int>(outputs.size());
            int ppb = n_out / yolo::NUM_BRANCHES;

            auto tp0 = std::chrono::steady_clock::now();

            for (int b = 0; b < yolo::NUM_BRANCHES; ++b) {
                const cv::Mat& box_t = outputs[b * ppb];
                const cv::Mat& cls_t = outputs[b * ppb + 1];

                int num_cls = cls_t.rows;
                int total_px = cls_t.cols;
                int h = static_cast<int>(std::sqrt(static_cast<float>(total_px)));
                int w = h;
                if (h * w != total_px) {
                    h = ih / (1 << (yolo::NUM_BRANCHES - 1 - b));
                    w = iw / (1 << (yolo::NUM_BRANCHES - 1 - b));
                }

                int mc = box_t.rows / 4;
                auto raw = yolo::decode_branch(box_t.ptr<float>(0), cls_t.ptr<float>(0),
                                               h, w, mc, num_cls, orig_h, orig_w,
                                               ai.confidence_threshold, ppbuf);
                all_raw.insert(all_raw.end(), raw.begin(), raw.end());
            }

            auto tp1 = std::chrono::steady_clock::now();

            auto boxes = yolo::nms_boxes(all_raw, ai.nms_threshold);

            auto tp2 = std::chrono::steady_clock::now();

            DetResult result;
            result.boxes    = std::move(boxes);
            result.frame_id = ++frame_counter;
            result.width    = orig_w;
            result.height   = orig_h;
            ctx_->det_result.publish(std::move(result));

            const auto& final_boxes = ctx_->det_result.get();
            if (final_boxes && !final_boxes->boxes.empty()) {
                auto best = std::max_element(
                    final_boxes->boxes.begin(), final_boxes->boxes.end(),
                    [](const DetectionBox& a, const DetectionBox& b) {
                        return a.y2 < b.y2;
                    });
                int cls = best->class_id;
                const char* scene = (cls >= 0 && cls < yolo::NUM_CLASSES)
                    ? yolo::CLASSES[cls] : "Unknown";
                ctx_->scene_manager.switchTo(scene);
            } else {
                ctx_->scene_manager.switchTo("NormalScene");
            }

            auto tp3 = std::chrono::steady_clock::now();
            auto dfl_us  = std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count();
            auto nms_us  = std::chrono::duration_cast<std::chrono::microseconds>(tp2 - tp1).count();
            auto pub_us  = std::chrono::duration_cast<std::chrono::microseconds>(tp3 - tp2).count();
            std::cout << "[Post] decode=" << dfl_us << "us  nms=" << nms_us
                      << "us  publish=" << pub_us << "us  raw=" << all_raw.size() << std::endl;
        }

        auto t3 = std::chrono::steady_clock::now();
        auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto post_us  = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        ctx_->perf_stats.recordStages(SLOT_DETECTION, 0, infer_us, post_us);
    }

    std::cout << "[DetTask] 已退出" << std::endl;
}
