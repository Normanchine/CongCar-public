#include "DetectionTask.h"
#include "../common/PerfStats.h"
#include "../common/SharedResult.h"
#include "../inference/RKNNEngine.h"
#include "../inference/PPYoloEPostProcess.h"

#include "../thread/ThreadTaskFactory.h"
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

REGISTER_THREAD_TASK(DetectionTask)

DetectionTask::DetectionTask(SharedContext* ctx) : IThreadTask(ctx) {}
DetectionTask::~DetectionTask() = default;

void DetectionTask::run() {
    RKNNEngine engine;
    const auto& det_cfg = ctx_->config.ai.detection;

    if (!engine.load(det_cfg.model_path, 0)) {
        std::cerr << "[DetTask] 模型加载失败" << std::endl;
        return;
    }

    const int iw = det_cfg.input_w;
    const int ih = det_cfg.input_h;

    std::cout << "[DetTask] 已启动 (输入: " << iw << "x" << ih << ")" << std::endl;

    // 流水线首帧预提交（CaptureTask 已完成 flip + resize，保持RGB）
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

        auto t_start = std::chrono::steady_clock::now();

        auto [outputs, infer_us] = future.get();

        future = engine.submit(std::move(next_input));

        auto t_post_start = std::chrono::steady_clock::now();

        if (outputs.size() >= 2) {
            int orig_w = ctx_->full_frame_width.load(std::memory_order_acquire);
            int orig_h = ctx_->full_frame_height.load(std::memory_order_acquire);

            // PP-YOLOE后处理：output[0]=cls(3×1680), output[1]=reg(1680×4)
            auto detections = ppyoloe::postprocess(
                outputs[0].ptr<float>(0), outputs[1].ptr<float>(0),
                orig_w, orig_h,
                orig_h, 0,    // 检测走全图，无额外裁剪
                det_cfg.confidence_threshold, det_cfg.nms_threshold);

            // 转换为DetResult（全图坐标）
            DetResult result;
            result.frame_id = ++frame_counter;
            result.width    = orig_w;
            result.height   = orig_h;
            result.boxes.reserve(detections.size());
            for (const auto& d : detections) {
                result.boxes.push_back({d.x1, d.y1, d.x2, d.y2, d.score, d.class_id});
            }
            ctx_->det_result.publish(std::move(result));

            // 场景切换：取图像最下方的目标，带防抖
            // 确定最佳检测对应的场景名
            std::string best_scene = "NormalScene";
            if (!detections.empty()) {
                auto best = std::max_element(detections.begin(), detections.end(),
                    [](const ppyoloe::Detection& a, const ppyoloe::Detection& b) {
                        return a.y2 < b.y2;
                    });
                int cls = best->class_id;
                const auto& classes = det_cfg.classes;
                if (cls >= 0 && cls < static_cast<int>(classes.size())) {
                    auto it = det_cfg.scene_map.find(classes[cls]);
                    if (it != det_cfg.scene_map.end())
                        best_scene = it->second;
                }
            }

            const auto& cur = ctx_->scene_manager.currentName();
            if (best_scene == cur) {
                det_lost_frames_ = 0;               // 持续命中，复位
            } else if (cur == "NormalScene") {
                ctx_->scene_manager.switchTo(best_scene);  // 从普通场景即时切入
            } else {
                det_lost_frames_++;                  // 特殊场景间切换需缓冲
                if (det_lost_frames_ >= det_cfg.lost_frame_limit) {
                    ctx_->scene_manager.switchTo(best_scene);
                    det_lost_frames_ = 0;
                }
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        auto post_us  = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_post_start).count();
        ctx_->perf_stats.recordStages(SLOT_DETECTION, total_us, infer_us, post_us);
    }

    std::cout << "[DetTask] 已退出" << std::endl;
}
