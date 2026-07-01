#include "TrafficSignScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include <algorithm>

//REGISTER_SCENE(TrafficSignScene)

// ---- OCR 函数：空实现，后续接入 API ----
int TrafficSignScene::recognizeTrafficSign(const cv::Mat& /*sign_image*/) {
    // TODO: 接入大模型 API，识别直走/右转
    // 返回值：0=直走，1=右转
    return 0;  // 默认直走
}

// ---- 重写 onFork：根据 OCR 结果选择保留侧 ----
void TrafficSignScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* /*ctx*/) {
    pathplan::handleFork(wide_mask, fork_side, 1);
}

void TrafficSignScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // 找到 traffic_sign 的 class_id（只查一次）
    if (traffic_sign_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "traffic_sign") {
                traffic_sign_class_id_ = i;
                break;
            }
        }
    }

    // ---- 1. 识别交通标志，设置分岔方向 ----
    if (traffic_sign_class_id_ >= 0) {
        auto det = ctx->det_result.get();
        if (det) {
            // 找置信度最高的交通标志
            const DetectionBox* best = nullptr;
            float best_score = 0.0f;
            for (const auto& b : det->boxes) {
                if (b.class_id == traffic_sign_class_id_ && b.confidence > best_score) {
                    best_score = b.confidence;
                    best = &b;
                }
            }

            if (best && best_score > 0.4f) {
                // 获取工作帧（与检测框同坐标系）
                cv::Mat work_frame;
                if (ctx->work_frame_queue.tryConsume(work_frame)) {
                    // 裁剪交通标志区域
                    int x1 = std::max(0, static_cast<int>(best->x1));
                    int y1 = std::max(0, static_cast<int>(best->y1));
                    int x2 = std::min(work_frame.cols, static_cast<int>(best->x2));
                    int y2 = std::min(work_frame.rows, static_cast<int>(best->y2));

                    if (x2 > x1 && y2 > y1) {
                        cv::Mat sign_roi = work_frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));

                        // 调用 OCR 识别
                        int direction = recognizeTrafficSign(sign_roi);
                        follower_.fork_side = direction;  // 0=左(直走)，1=右(右转)
                    }
                }
            }
        }
    }

    // ---- 2. 正常巡线（PathFollower 会调用重写的 onFork） ----
    follower_.process(seg->mask, ctx);
}
