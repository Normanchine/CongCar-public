#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include <algorithm>
#include <cmath>
#include <iostream>

REGISTER_SCENE(PersonScene)

#define DEG2RAD (static_cast<float>(M_PI) / 180.0f)

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
// APPROACH：正常巡线 + 人到鸟瞰图纵向中点 → 触发
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
        if (person_class_id_ < 0) return;
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

    const auto& crop = ctx->config.frame_crop;
    if (crop.enable) {
        person_img_x -= static_cast<float>(crop.x);
        person_img_y -= static_cast<float>(crop.y);
    }

    cv::Point2f person_bird = ctx->perspective.toBird(
        cv::Point2f(person_img_x, person_img_y));
    const auto& calib = ctx->config.calib;

    // ---- 5. 触发判定：人进入鸟瞰图纵向中点 ----
    float half_h = static_cast<float>(calib.bird_height) * 0.5f;
    if (person_bird.y > half_h) return;  // 人还在下半区，不触发

    // ---- 6. 触发！生成引导线 ----
    cv::Mat wide_mask = ctx->perspective.warp(seg_mask);
    if (!generateGuideLine(wide_mask, person_bird, ctx)) {
        std::cerr << "[PersonScene] 引导线生成失败" << std::endl;
        return;
    }

    // ---- 7. 进入 HOLDING：初始化车在引导线坐标系上的起点 ----
    ctx->scene_manager.hold();
    const auto& ctrl = ctx->config.control;
    car_x_ = static_cast<float>(calib.bird_width) * calib.bird_width_scale * 0.5f;
    car_y_ = static_cast<float>(calib.bird_height) * ctrl.car_cy_ratio;
    car_x_init_ = car_x_;
    car_y_init_ = car_y_;
    entry_yaw_  = ctx->yaw.load(std::memory_order_acquire);
    rel_yaw_    = 0.0f;
    last_tp_    = std::chrono::steady_clock::now();
    segment1_first_ = true;
    segment2_first_ = true;
    state_      = State::HOLDING;

    std::cout << "[PersonScene] 触发绕行！car=(" << car_x_ << "," << car_y_ << ")"
              << ", person.y=" << person_bird.y << " (half=" << half_h << ")"
              << ", M=(" << M_.x << "," << M_.y << ")"
              << ", E=(" << E_.x << "," << E_.y << ")" << std::endl;
}

// ============================================================================
// HOLDING：引导线固定，车由 yaw 驱动在坐标系上移动
//   段1（car_y_ > M_.y）：锁 M → 快速建立横向分离
//   段2（car_y_ ≤ M_.y）：锁 E → 平滑汇入中线
//   退出（car_y_ ≤ E_.y）：车越过终点
// ============================================================================

void PersonScene::handleHolding(SharedContext* ctx) {
    // ---- 0. IMU 相对航向 ----
    rel_yaw_ = (ctx->yaw.load(std::memory_order_acquire) - entry_yaw_)
             * ctx->config.control.yaw_direction;
    while (rel_yaw_ > 180.0f)  rel_yaw_ -= 360.0f;
    while (rel_yaw_ < -180.0f) rel_yaw_ += 360.0f;

    // ---- 1. 车的位置更新：由 yaw 驱动 ----
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - last_tp_).count();
    last_tp_ = now;
    if (dt > 0.1f) dt = 0.1f;

    float speed = ctx->target_speed.load(std::memory_order_acquire);
    float rad   = rel_yaw_ * DEG2RAD;
    float dist  = speed * dt;   // 轮子滚过的距离 (m)

    const auto& ctrl  = ctx->config.control;
    const auto& calib = ctx->config.calib;
    const auto& sp    = ctrl.scene.at("PersonScene");

    car_y_ -= dist * calib.ppm_y * std::cos(rad);  // 纵向（车前进 = Y↓）
    car_x_ += dist * calib.ppm_x * std::sin(rad);  // 横向（右转为正）

    // ---- 2. 确定当前段 + 目标点（引导线固定不动） ----
    cv::Point2f target;
    bool segment1 = false;
    bool skip_d   = false;

    if (car_y_ > M_.y) {
        // 段1：M 还在前方 → 锁 M
        target   = M_;
        segment1 = true;
    } else if (car_y_ > E_.y) {
        // 段2：已过 M，E 还在前方 → 锁 E
        target   = E_;
        segment1 = false;
    } else {
        // E 也过了 → 到达终点
        state_ = State::DONE;
        return;
    }

    // ---- 3. 首帧 / 切段首帧：继承 ctx 状态 + 跳过 D 项 ----
    if (segment1 && segment1_first_) {
        last_error_ = ctx->last_error.load(std::memory_order_acquire);
        last_speed_ = ctx->target_speed.load(std::memory_order_acquire);
        segment1_first_ = false;
        skip_d = true;
    }
    if (!segment1 && segment2_first_) {
        segment2_first_ = false;
        skip_d = true;
    }

    // ---- 4. 航向误差：headingError 假设车朝正前方，补偿 yaw ----
    float error = pathplan::headingError(car_x_, car_y_,
                                          target.x, target.y,
                                          ctrl.car_length, calib.ppm_x, calib.ppm_y)
                * ctx->steering_dir
                - rad;  // 补偿已转角度

    // ---- 5. PD ----
    float derivative = skip_d ? 0.0f : (error - last_error_);
    last_error_ = error;

    float pwm = ctrl.pwm_mid - (sp.pid_p * error + sp.pid_d * derivative) * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);

    // ---- 6. 速度：大弯减速 + 加速平滑恢复 ----
    float target_spd = sp.max_speed;
    float ae = std::abs(error);
    if (ae > ctrl.steer_speed_threshold) {
        float excess = ae - ctrl.steer_speed_threshold;
        target_spd = sp.max_speed / (1.0f + ctrl.steer_speed_k * excess);
    }
    if (target_spd < last_speed_) last_speed_ = target_spd;
    else last_speed_ += (target_spd - last_speed_) * sp.speed_recovery_k;

    // ---- 7. 写回共享状态 + 更新引导线地图 ----
    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(last_speed_);
    ctx->last_error.store(last_error_, std::memory_order_release);

    // 引导线固定不动，不滚动
    {
        std::lock_guard<std::mutex> lk(ctx->guide_mtx_);
        ctx->guide_mask_   = guide_mask_.clone();
        ctx->guide_pts_    = guide_pts_;
        ctx->guide_M_      = M_;
        ctx->guide_E_      = E_;
        ctx->guide_target_ = target;
        ctx->guide_valid_  = true;
        // 写入全局坐标系元数据
        ctx->guide_meta_.origin  = {car_x_init_, car_y_init_};  // 记录入口位置
        ctx->guide_meta_.e_pt    = E_;
        ctx->guide_meta_.car_pt  = cv::Point2f(car_x_, car_y_);
        ctx->guide_meta_.car_yaw = rel_yaw_;
        ctx->guide_meta_.blind   = false;
    }

    // 同时写一份到 bird_centers_ 给右侧鸟瞰图显示
    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->bird_mtx_);
        ctx->bird_centers_ = guide_pts_;
        ctx->bird_aim_pt_  = target;
    }
}

// ============================================================================
// DONE：释放 hold，切回 NormalScene
// ============================================================================

void PersonScene::handleDone(SharedContext* ctx) {
    ctx->scene_manager.release();
    ctx->scene_manager.switchTo("NormalScene");
    {
        std::lock_guard<std::mutex> lk(ctx->guide_mtx_);
        ctx->guide_valid_ = false;
    }
    std::cout << "[PersonScene] 绕行完成 → NormalScene"
              << " (rel_yaw=" << rel_yaw_ << "°"
              << ", car=(" << car_x_ << "," << car_y_ << "))" << std::endl;
}

// ============================================================================
// generateGuideLine：中线逐点等比平移 → 引导线
// ============================================================================

bool PersonScene::generateGuideLine(const cv::Mat& wide_mask,
                                     const cv::Point2f& person_bird,
                                     SharedContext* ctx) {
    const auto& calib = ctx->config.calib;
    const auto& sp    = ctx->config.control.scene.at("PersonScene");

    int bird_w = wide_mask.cols;
    int bird_h = wide_mask.rows;
    float ppm_x = calib.ppm_x;
    float ppm_y = calib.ppm_y;

    // ---- 1. 提取中线 + 平滑 ----
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(wide_mask, centers);
    if (centers.empty()) return false;

    pathplan::smoothCenterline(centers);
    // centers: 从远到近（Y 递增），centers[0]=最远(最小Y)，centers.back()=最近(最大Y)

    // ---- 2. 中线插值（按 Y 查找对应 X） ----
    auto interpolateX = [&](float y) -> float {
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

    // ---- 3. 三个基准点的纵坐标 ----
    float car_cy = static_cast<float>(bird_h) * ctx->config.control.car_cy_ratio;
    float y_S = car_cy;
    float y_M = person_bird.y;
    float y_E = centers[0].y;  // 中线最远点（最小Y）

    // ---- 4. M 在中线上的原始 X + 绕行方向 ----
    float x_M = interpolateX(y_M);

    float lateral_sign;
    if (person_bird.x < x_M)       lateral_sign = 1.0f;   // 人在左→往右绕
    else if (person_bird.x > x_M)  lateral_sign = -1.0f;  // 人在右→往左绕
    else                            lateral_sign = 1.0f;   // 压中线，默认右绕

    float offset_M_px = sp.person_side_offset_m * ppm_x;

    // ---- 5. 逐点等比平移 ----
    guide_pts_.clear();
    guide_pts_.reserve(centers.size());

    float denom1 = y_S - y_M;  // 段1纵向跨度
    float denom2 = y_M - y_E;  // 段2纵向跨度

    for (const auto& p : centers) {
        float offset = 0.0f;
        if (p.y >= y_M && p.y <= y_S && denom1 > 1e-4f) {
            // 段1（S→M）：偏移 0 → offset_M，线性增长
            offset = offset_M_px * (y_S - p.y) / denom1;
        } else if (p.y >= y_E && p.y < y_M && denom2 > 1e-4f) {
            // 段2（M→E）：偏移 offset_M → 0，线性衰减
            offset = offset_M_px * (p.y - y_E) / denom2;
        }
        // else: y > y_S（车后）或 y < y_E → offset=0

        guide_pts_.push_back(cv::Point2f(p.x + lateral_sign * offset, p.y));
    }

    // ---- 6. 记录 M 和 E（偏移后）+ 保存底图供DisplayTask ----
    M_ = cv::Point2f(x_M + lateral_sign * offset_M_px, y_M);
    E_ = cv::Point2f(centers[0].x, y_E);  // E 偏移=0，回归中线
    guide_mask_ = wide_mask.clone();

    return true;
}
