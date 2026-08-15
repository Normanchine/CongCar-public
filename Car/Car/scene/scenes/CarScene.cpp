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

    // ---- 每帧：找 car 框 → 写障碍鸟瞰点（供 DisplayTask 画点） ----
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

        // 写共享障碍点（宽鸟瞰坐标）
        {
            std::lock_guard<std::mutex> lk(ctx->car_obs_mtx_);
            ctx->car_obs_pt_    = obs;
            ctx->car_obs_valid_ = true;
        }

        // ---- 实时检测：每帧判断车在赛道哪边 + 切换边界 ----
        // 赛道中心基准：在障碍 y 处插值出赛道中线 x（不是画面中心！）
        float track_center_x = trackCenterXAt(seg->mask, obs.y, ctx);

        // 障碍在中线左 → 沿右边界(+1)；障碍在中线右 → 沿左边界(-1)
        int edge = (obs.x < track_center_x) ? 1 : -1;
        follower_.setFollowEdge(edge);

        ctx->car_edge.store(edge, std::memory_order_release);
        // 边界是 Car 场景局部：setFollowEdge 已切，退出场景即作废，不写全局（Normal 固定左边界）
    } else {
        // 本帧无 car 框：清掉障碍点标记
        std::lock_guard<std::mutex> lk(ctx->car_obs_mtx_);
        ctx->car_obs_valid_ = false;
    }

    // ---- 沿指定边界巡线（全程宽鸟瞰） ----
    follower_.process(seg->mask, ctx);
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

// 宽鸟瞰赛道中线在 bird_y 处的 x（插值）；中线为空用宽鸟瞰中心兜底
float CarScene::trackCenterXAt(const cv::Mat& seg_mask, float bird_y, SharedContext* ctx) {
    cv::Mat wide_mask = ctx->perspective.warp(seg_mask);
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(wide_mask, centers);   // 取中点=赛道中线
    if (centers.empty()) return static_cast<float>(wide_mask.cols) * 0.5f;
    pathplan::smoothCenterline(centers);

    // 按 y 插值 x（centers 从远到近，Y 递增）
    for (size_t i = 0; i + 1 < centers.size(); ++i) {
        float y1 = centers[i].y;
        float y2 = centers[i + 1].y;
        if ((bird_y <= y1 && bird_y >= y2) || (bird_y >= y1 && bird_y <= y2)) {
            if (std::abs(y2 - y1) < 1e-4f) return centers[i].x;
            float t = (bird_y - y1) / (y2 - y1);
            return centers[i].x + t * (centers[i + 1].x - centers[i].x);
        }
    }

    // 兜底：最近点
    float best_d = 1e9f, best_x = static_cast<float>(wide_mask.cols) * 0.5f;
    for (const auto& p : centers) {
        float d = std::abs(p.y - bird_y);
        if (d < best_d) { best_d = d; best_x = p.x; }
    }
    return best_x;
}
