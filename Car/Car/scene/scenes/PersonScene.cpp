#include "PersonScene.h"
#include "../../hal/uart/UartIO.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include <cmath>
#include <cstdint>
#include <iostream>

REGISTER_SCENE(PersonScene)

// 行人丢失多少帧后恢复巡线（防检测单帧抖动导致前冲）
static constexpr int kLostPersonResumeFrames = 5;

void PersonScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // ---- 1. 找 person 框 → 底部投影 → 十字显示 ----
    if (person_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "person") { person_class_id_ = i; break; }
        }
        if (person_class_id_ < 0) {
            std::cerr << "[PersonScene] 未找到 person 类别！" << std::endl;
            return;
        }
    }

    DetectionBox box;
    bool has_person = findPersonBox(ctx, box);

    if (has_person) {
        lost_person_frames_ = 0;
        cv::Point2f obs = boxToBird(box, ctx);

        // 写共享十字显示
        {
            std::lock_guard<std::mutex> lk(ctx->person_obs_mtx_);
            ctx->person_obs_pt_    = obs;
            ctx->person_obs_valid_ = true;
        }

        // 更新运动方向 + 赛道中心基准
        updateMotion(obs);
        float track_center_x = trackCenterXAt(seg->mask, obs.y, ctx);

        // ---- 停车：行人底部到达停止比例（鸟瞰图高度的 person_stop_ratio） ----
        const auto& sp = ctx->config.control.scene.at(ctx->scene_manager.currentName());
        float stop_y = static_cast<float>(ctx->perspective.birdH()) * sp.person_stop_ratio;
        if (obs.y >= stop_y) stopped_ = true;

        // ---- 起步：人让开路（当前边界 + 方向 + 已过赛道中心） ----
        if (stopped_ && shouldGo(obs, track_center_x)) stopped_ = false;
    } else {
        // 无 person：清显示；短暂丢失保持停车，防检测抖动前冲
        {
            std::lock_guard<std::mutex> lk(ctx->person_obs_mtx_);
            ctx->person_obs_valid_ = false;
        }
        if (++lost_person_frames_ > kLostPersonResumeFrames) {
            stopped_  = false;
            has_prev_ = false;
        }
    }

    // ---- 2. 停车 or 巡线 ----
    if (stopped_) {
        // 不溜车：场景内直发停车指令（绕过 ControlTask 下发循环）
        const uint16_t neutral = static_cast<uint16_t>(ctx->config.uart.pwm_neutral);
        if (ctx->uart_io) ctx->uart_io->sendControl(0.0f, neutral);
        ctx->target_speed.store(0.0f);
        ctx->steering_pwm.store(static_cast<float>(neutral));
    } else {
        follower_.process(seg->mask, ctx);
    }
}

// ============================================================================
// 辅助
// ============================================================================

bool PersonScene::findPersonBox(SharedContext* ctx, DetectionBox& out) {
    if (person_class_id_ < 0) return false;
    auto det = ctx->det_result.get();   // shared_ptr 保活，避免返回悬垂指针
    if (!det || det->boxes.empty()) return false;

    const DetectionBox* best = nullptr;
    float best_conf = 0.0f;
    for (const auto& b : det->boxes) {
        if (b.class_id == person_class_id_ && b.confidence > best_conf) {
            best_conf = b.confidence;
            best = &b;
        }
    }
    if (!best) return false;
    out = *best;   // 拷出副本
    return true;
}

// 框底边中点（全图坐标）→ 减 frame_crop → 鸟瞰坐标
cv::Point2f PersonScene::boxToBird(const DetectionBox& box, SharedContext* ctx) {
    float img_x = (box.x1 + box.x2) * 0.5f;
    float img_y = box.y2;   // 底部中点：脚部地面接触点（框中心不在地面，投影会偏近）

    const auto& crop = ctx->config.frame_crop;
    if (crop.enable) {
        img_x -= static_cast<float>(crop.x);
        img_y -= static_cast<float>(crop.y);
    }
    return ctx->perspective.toBird(cv::Point2f(img_x, img_y));
}

// 宽鸟瞰赛道中线在 bird_y 处的 x（插值）；中线为空用宽鸟瞰中心兜底
float PersonScene::trackCenterXAt(const cv::Mat& seg_mask, float bird_y, SharedContext* ctx) {
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

// 更新行人左右移动方向（逐帧 x 变化）
void PersonScene::updateMotion(const cv::Point2f& obs) {
    if (!has_prev_) {
        prev_obs_ = obs;
        has_prev_ = true;
        moving_right_ = moving_left_ = false;
        return;
    }
    float dx = obs.x - prev_obs_.x;
    prev_obs_ = obs;

    // 方向阈值（鸟瞰像素），避免单帧抖动误判
    constexpr float kDirThresholdPx = 3.0f;
    if (dx > kDirThresholdPx) {
        moving_right_ = true;
        moving_left_  = false;
    } else if (dx < -kDirThresholdPx) {
        moving_left_  = true;
        moving_right_ = false;
    }
    // 阈值内保持上一方向状态
}

// 起步判断：按当前边界 + 行人方向 + 已过赛道中心
bool PersonScene::shouldGo(const cv::Point2f& obs, float track_center_x) const {
    int edge = follower_.getFollowEdge();
    if (edge < 0) {
        // 左边界：人向右走 且 人已过赛道中心（x > 中心 x）
        return moving_right_ && obs.x > track_center_x;
    }
    // 右边界：人向左走 且 人已过赛道中心（x < 中心 x）
    return moving_left_ && obs.x < track_center_x;
}
