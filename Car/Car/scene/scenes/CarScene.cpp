#include "CarScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include <iostream>

REGISTER_SCENE(CarScene)

// ============================================================================
// 主循环
// ============================================================================

void CarScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // 宽鸟瞰巡线：沿边界走时远处边界不脱离视野
    follower_.setCropScale(ctx->config.calib.bird_width_scale);

    // ---- 首次进入：锁存沿边方向（只一次） ----
    if (!edge_locked_) {
        if (car_class_id_ < 0) {
            const auto& classes = ctx->config.ai.detection.classes;
            for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
                if (classes[i] == "car") { car_class_id_ = i; break; }
            }
            if (car_class_id_ < 0) {
                std::cerr << "[CarScene] 未找到 car 类别！" << std::endl;
                return;
            }
        }

        DetectionBox box;
        if (findCarBox(ctx, box)) {
            cv::Point2f obs = boxToBird(box, ctx);
            const auto& calib = ctx->config.calib;
            float wide_w = static_cast<float>(calib.bird_width) * calib.bird_width_scale;

            // 障碍在中线左 → 沿右边界(+1)；障碍在中线右 → 沿左边界(-1)
            follower_.edge = (obs.x < wide_w * 0.5f) ? 1 : -1;
            edge_locked_   = true;

            ctx->car_edge.store(follower_.edge, std::memory_order_release);
            std::cout << "[CarScene] 沿" << (follower_.edge > 0 ? "右" : "左")
                      << "边界巡线 (障碍x=" << obs.x << ", 中线=" << wide_w * 0.5f << ")"
                      << std::endl;
        }
        // 未锁存前沿中线（edge=0），DetTask 检测到 car 后会重新切入
    }

    // ---- 沿指定边界巡线 ----
    follower_.process(seg_mask, ctx);
}

// ============================================================================
// Follower：onCenterline 沿指定边界提取（不取中点）
// ============================================================================

void CarScene::Follower::onCenterline(const cv::Mat& mask,
                                      std::vector<cv::Point2f>& centers,
                                      SharedContext* /*ctx*/) {
    pathplan::extractBoundary(mask, centers, edge);
    pathplan::smoothCenterline(centers);
}

// ============================================================================
// 辅助
// ============================================================================

bool CarScene::findCarBox(SharedContext* ctx, DetectionBox& out) {
    if (car_class_id_ < 0) return false;
    auto det = ctx->det_result.get();   // shared_ptr 保活，避免返回悬垂指针
    if (!det || det->boxes.empty()) return false;

    const DetectionBox* best = nullptr;
    float best_conf = 0.0f;
    for (const auto& b : det->boxes) {
        if (b.class_id == car_class_id_ && b.confidence > best_conf) {
            best_conf = b.confidence;
            best = &b;
        }
    }
    if (!best) return false;
    out = *best;   // 拷出副本
    return true;
}

// 框底边中点（全图坐标）→ 减 frame_crop → 鸟瞰坐标
cv::Point2f CarScene::boxToBird(const DetectionBox& box, SharedContext* ctx) {
    float img_x = (box.x1 + box.x2) * 0.5f;
    float img_y = box.y2;   // 底边中点：地面接触点（框中心不在地面，投影会偏近）

    const auto& crop = ctx->config.frame_crop;
    if (crop.enable) {
        img_x -= static_cast<float>(crop.x);
        img_y -= static_cast<float>(crop.y);
    }
    return ctx->perspective.toBird(cv::Point2f(img_x, img_y));
}
