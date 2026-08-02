#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include <algorithm>
#include <cmath>
#include <iostream>

REGISTER_SCENE(PersonScene)

// ============================================================================
// run() — 状态机入口
// ============================================================================

void PersonScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    switch (state_) {
        case State::APPROACH: handleApproach(ctx, seg->mask); break;
        case State::HOLDING:  handleHolding(ctx);             break;
        case State::DONE:     handleDone(ctx);                break;
    }
}

// ============================================================================
// APPROACH：正常巡线 + 距离判定 → 触发生成弧线
// ============================================================================

void PersonScene::handleApproach(SharedContext* ctx, const cv::Mat& seg_mask) {
    // ---- 1. 正常巡线 ----
    follower_.process(seg_mask, ctx);

    // ---- 2. 缓存 person class_id（首次） ----
    if (person_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "person") { person_class_id_ = i; break; }
        }
        if (person_class_id_ < 0) return;  // 检测模型未配置 person 类别，保持 APPROACH
    }

    // ---- 3. 从检测结果找置信度最高的人框 ----
    auto det = ctx->det_result.get();
    if (!det || det->boxes.empty()) return;

    const DetectionBox* best = nullptr;
    float best_conf = 0.0f;
    for (const auto& b : det->boxes) {
        if (b.class_id == person_class_id_ && b.confidence > best_conf) {
            best_conf = b.confidence;
            best = &b;
        }
    }
    if (!best) return;

    // ---- 4. 人框底部中点 → 鸟瞰坐标 ----
    float person_img_x = (best->x1 + best->x2) * 0.5f;
    float person_img_y = best->y2;

    // 若帧裁剪了，检测框在全图空间，需偏移到裁剪空间
    const auto& crop = ctx->config.frame_crop;
    if (crop.enable) {
        person_img_x -= static_cast<float>(crop.x);
        person_img_y -= static_cast<float>(crop.y);
    }

    cv::Point2f person_bird = ctx->perspective.toBird(cv::Point2f(person_img_x, person_img_y));
    const auto& calib = ctx->config.calib;

    float car_cy = static_cast<float>(calib.bird_height) * ctx->config.control.car_cy_ratio;
    float dist_m = (car_cy - person_bird.y) / calib.ppm_y;

    // ---- 5. 纵向距离判定 ----
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    if (dist_m > sp.person_critical_dist_m) return;

    // ---- 6. 触发生成弧线 ----
    cv::Mat wide_mask = ctx->perspective.warp(seg_mask);
    if (!generateArc(wide_mask, person_bird, ctx)) {
        std::cerr << "[PersonScene] 弧线生成失败" << std::endl;
        return;
    }

    // ---- 7. 进入 HOLDING ----
    ctx->scene_manager.hold();
    traveled_m_ = 0.0f;
    last_tp_    = std::chrono::steady_clock::now();
    state_      = State::HOLDING;

    std::cout << "[PersonScene] 触发绕行！dist=" << dist_m
              << "m, arc_len=" << arc_total_length_m_ << "m" << std::endl;
}

// ============================================================================
// HOLDING：每帧移位弧线 → 逆变换回图像 → 巡线 + 航位推算
// ============================================================================

void PersonScene::handleHolding(SharedContext* ctx) {
    // ---- 1. 航位推算 ----
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_tp_).count();
    last_tp_ = now;

    float speed = ctx->target_speed.load(std::memory_order_acquire);
    traveled_m_ += speed * dt;

    // ---- 2. 弧线走完？ ----
    if (traveled_m_ >= arc_total_length_m_) {
        state_ = State::DONE;
        return;
    }

    // ---- 3. 弧线纵向移位 + 重绘到鸟瞰 mask ----
    const auto& calib = ctx->config.calib;
    int bird_h = calib.bird_height;
    int bird_w = static_cast<int>(calib.bird_width * calib.bird_width_scale);
    float shift_y = traveled_m_ * calib.ppm_y;

    cv::Mat bird_mask = cv::Mat::zeros(bird_h, bird_w, CV_8UC1);

    std::vector<cv::Point> shifted_pts;
    shifted_pts.reserve(arc_bezier_pts_.size());
    for (const auto& p : arc_bezier_pts_) {
        shifted_pts.push_back(cv::Point(
            static_cast<int>(p.x),
            static_cast<int>(p.y + shift_y)));
    }

    // 逐段画粗线（cv::line 自动裁剪到图内）
    for (size_t i = 1; i < shifted_pts.size(); ++i) {
        cv::line(bird_mask, shifted_pts[i - 1], shifted_pts[i], 255, arc_thickness_px_);
    }

    // ---- 4. 逆变换到图像坐标 → 巡线 ----
    int seg_w = ctx->frame_width.load(std::memory_order_acquire);
    int seg_h = ctx->frame_height.load(std::memory_order_acquire);
    if (seg_w <= 0 || seg_h <= 0) return;  // CaptureTask 还没写入尺寸

    cv::Mat arc_img = ctx->perspective.warpInverse(bird_mask, seg_w, seg_h);
    follower_.process(arc_img, ctx);
}

// ============================================================================
// DONE：释放 hold，切回正常场景
// ============================================================================

void PersonScene::handleDone(SharedContext* ctx) {
    ctx->scene_manager.release();
    ctx->scene_manager.switchTo("NormalScene");
    std::cout << "[PersonScene] 弧线绕行完成，切回 NormalScene" << std::endl;
}

// ============================================================================
// generateArc：在鸟瞰图坐标系生成贝塞尔弧线
// ============================================================================

bool PersonScene::generateArc(const cv::Mat& wide_mask,
                               const cv::Point2f& person_bird,
                               SharedContext* ctx) {
    const auto& calib = ctx->config.calib;
    const auto& sp    = ctx->config.control.scene.at("PersonScene");

    int bird_w = wide_mask.cols;
    int bird_h = wide_mask.rows;
    float ppm_x = calib.ppm_x;
    float ppm_y = calib.ppm_y;

    // ---- 1. 在 wide_mask 上提取中线 ----
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(wide_mask, centers);
    if (centers.empty()) return false;

    pathplan::smoothCenterline(centers);

    // ---- 2. 中线插值辅助函数 ----
    // centers: 从底（近车）到顶（远处），Y 递减
    auto interpolateX = [&](float y) -> float {
        // 找 Y 跨过 y 的相邻两点线性插值
        for (size_t i = 0; i + 1 < centers.size(); ++i) {
            float y1 = centers[i].y;
            float y2 = centers[i + 1].y;
            if ((y <= y1 && y >= y2) || (y >= y1 && y <= y2)) {
                if (std::abs(y2 - y1) < 1e-4f) return centers[i].x;
                float t = (y - y1) / (y2 - y1);
                return centers[i].x + t * (centers[i + 1].x - centers[i].x);
            }
        }
        // 兜底：最近点
        float best_d = 1e9f, best_x = static_cast<float>(bird_w) * 0.5f;
        for (const auto& p : centers) {
            float d = std::abs(p.y - y);
            if (d < best_d) { best_d = d; best_x = p.x; }
        }
        return best_x;
    };

    // ---- 3. 弧线起止点 ----
    float car_cy = static_cast<float>(bird_h) * ctx->config.control.car_cy_ratio;

    float p0_y = car_cy - sp.person_arc_takeoff_dist_m * ppm_y;
    float p3_y = person_bird.y - sp.person_arc_landing_dist_m * ppm_y;

    float p0_x = interpolateX(p0_y);
    float p3_x = interpolateX(p3_y);

    cv::Point2f P0(p0_x, p0_y);
    cv::Point2f P3(p3_x, p3_y);

    // ---- 4. 赛道宽度（在 person_bird.y 行扫描） ----
    int scan_y = static_cast<int>(person_bird.y);
    scan_y = std::clamp(scan_y, 0, bird_h - 1);

    float mid_x = interpolateX(static_cast<float>(scan_y));
    int cx = static_cast<int>(mid_x);
    int left = cx, right = cx;
    while (left > 0 && wide_mask.at<uint8_t>(scan_y, left) > 0) --left;
    while (right < bird_w - 1 && wide_mask.at<uint8_t>(scan_y, right) > 0) ++right;

    float track_width_px = static_cast<float>(right - left);
    if (track_width_px < 5.0f) track_width_px = 40.0f;  // 兜底最小宽度

    // ---- 5. 绕行侧：人在中线哪侧，反侧绕 ----
    // lateral_sign: +1 = 往右绕（X+方向），-1 = 往左绕（X-方向）
    float lateral_sign = (person_bird.x < mid_x) ? -1.0f : 1.0f;

    float offset_px = track_width_px * sp.person_arc_offset_ratio;

    // ---- 6. 贝塞尔四点 ----
    float dY = std::abs(P3.y - P0.y);
    if (dY < 1.0f) dY = 1.0f;

    // P1：从 P0 出弯，大偏移（快速避开人）
    cv::Point2f dir_forward(0.0f, -dY * 0.35f);
    cv::Point2f lateral_out(lateral_sign * offset_px * 0.8f, 0.0f);
    cv::Point2f P1 = P0 + dir_forward + lateral_out;

    // P2：从外侧斜切入 P3，小偏移（浅角汇入，车不会横着进赛道）
    cv::Point2f dir_backward(0.0f, dY * 0.40f);
    cv::Point2f lateral_in(lateral_sign * offset_px * 0.25f, 0.0f);
    cv::Point2f P2 = P3 + dir_backward + lateral_in;

    // ---- 7. 采样贝塞尔（100个点） ----
    arc_bezier_pts_.clear();
    arc_bezier_pts_.reserve(101);
    for (int i = 0; i <= 100; ++i) {
        float t = static_cast<float>(i) / 100.0f;
        float u = 1.0f - t;
        float x = u * u * u * P0.x + 3.0f * u * u * t * P1.x
                + 3.0f * u * t * t * P2.x + t * t * t * P3.x;
        float y = u * u * u * P0.y + 3.0f * u * u * t * P1.y
                + 3.0f * u * t * t * P2.y + t * t * t * P3.y;
        arc_bezier_pts_.push_back(cv::Point2f(x, y));
    }

    // ---- 8. 计算弧线总长（米） ----
    arc_total_length_m_ = 0.0f;
    for (size_t i = 1; i < arc_bezier_pts_.size(); ++i) {
        float dx = (arc_bezier_pts_[i].x - arc_bezier_pts_[i - 1].x) / ppm_x;
        float dy = (arc_bezier_pts_[i].y - arc_bezier_pts_[i - 1].y) / ppm_y;
        arc_total_length_m_ += std::sqrt(dx * dx + dy * dy);
    }

    // ---- 9. 弧线绘制宽度 = 原赛道宽度 ----
    arc_thickness_px_ = std::max(5, static_cast<int>(track_width_px));

    return true;
}
