#include "PathFollower.h"
#include "../../thread/SharedContext.h"
#include <algorithm>
#include <cmath>
#include <iostream>

#define DEG2RAD (static_cast<float>(M_PI) / 180.0f)

// ---- 模板方法：固定流程 ----
bool PathFollower::process(const cv::Mat& seg_mask, SharedContext* ctx) {
    const auto& calib = ctx->config.calib;
    const auto& ctrl  = ctx->config.control;

    // 首帧从ctx继承error和speed，场景切换无脉冲
    if (first_frame_) {
        last_error_ = ctx->last_error.load(std::memory_order_acquire);
        last_speed_ = ctx->target_speed.load(std::memory_order_acquire);
        entry_yaw_  = ctx->yaw.load(std::memory_order_acquire);
        last_tp_    = std::chrono::steady_clock::now();
    }

    // ---- 0. 宽鸟瞰变换 ----
    cv::Mat wide_mask = ctx->perspective.warp(seg_mask);

    // ---- 1. 岔路处理（可重写） ----
    onFork(wide_mask, ctx);

    // ---- 2. 裁中心区域供路径规划 ----
    int crop_w = calib.bird_width;
    int offset_x = (wide_mask.cols - crop_w) / 2;
    cv::Mat mask = wide_mask(cv::Rect(offset_x, 0, crop_w, wide_mask.rows)).clone();

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->bird_mtx_);
        ctx->bird_mask_  = mask.clone();
        ctx->bird_valid_ = true;
    }

    // ---- 3. 中线提取（可重写） ----
    std::vector<cv::Point2f> centers;
    onCenterline(mask, centers, ctx);

    // ---- 3.5 全局坐标系更新 ----
    // centers 在裁切坐标系，转为宽鸟瞰坐标系后再写入 guide_meta
    {
        auto now = std::chrono::steady_clock::now();
        float dt = first_frame_ ? 0.0f
                   : std::chrono::duration<float>(now - last_tp_).count();
        last_tp_ = now;
        if (dt > 0.1f) dt = 0.1f;

        float yaw     = ctx->yaw.load(std::memory_order_acquire);
        float rel_yaw = (yaw - entry_yaw_) * ctrl.yaw_direction;
        while (rel_yaw > 180.0f)  rel_yaw -= 360.0f;
        while (rel_yaw < -180.0f) rel_yaw += 360.0f;
        float rad = rel_yaw * DEG2RAD;

        float birdW  = calib.bird_width * calib.bird_width_scale;
        float birdH  = static_cast<float>(calib.bird_height);
        float car_cx = birdW * 0.5f;
        float car_cy = birdH * ctrl.car_cy_ratio;

        float speed = (last_speed_ > 0.01f) ? last_speed_ : 0.0f;

        std::lock_guard<std::mutex> lk(ctx->guide_mtx_);
        auto& meta = ctx->guide_meta_;

        // yaw 驱动车位更新
        if (dt > 0.0f) {
            float dist = speed * dt;
            meta.car_pt.x += dist * calib.ppm_x * std::sin(rad);
            meta.car_pt.y -= dist * calib.ppm_y * std::cos(rad);
        }
        meta.car_yaw = rel_yaw;

        if (!centers.empty()) {
            // 赛道可见 → 重置坐标系
            meta.origin = {car_cx, car_cy};
            meta.e_pt   = {centers[0].x + static_cast<float>(offset_x),
                           centers[0].y};
            meta.car_pt = {car_cx, car_cy};
            meta.blind  = false;
            entry_yaw_  = yaw;  // 刷新 yaw 基准
        } else {
            // 盲开 → 冻结 origin + e_pt，只更新 car_pt
            meta.blind = true;
        }
    }

    // ---- 4. 丢线容错（可重写） ----
    if (centers.empty()) {
        cv::Point2f e_pt;
        float yaw_rad;
        {
            std::lock_guard<std::mutex> lk(ctx->guide_mtx_);
            e_pt    = ctx->guide_meta_.e_pt;
            yaw_rad = ctx->guide_meta_.car_yaw * DEG2RAD;
        }
        onLost(e_pt, yaw_rad, ctx);
        return true;
    }
    lost_frames_ = 0;

    // ---- 5. 纯跟踪 + 航向误差（可重写） ----
    cv::Point2f aim_pt;
    float error;
    if (!onPurePursuit(mask, centers, error, aim_pt, ctx))
        return true;

    // ---- 6. PD + PWM（可重写） ----
    float pwm = onPD(error, ctx);
    last_pwm_ = pwm;

    // ---- 7. 速度控制（可重写） ----
    float speed = onSpeed(error, ctx);
    last_speed_ = speed;

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(speed);
    ctx->last_error.store(last_error_, std::memory_order_release);
    first_frame_ = false;

    // ---- 8. debug 鸟瞰图更新 ----
    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->bird_mtx_);
        ctx->bird_centers_ = centers;
        ctx->bird_aim_pt_  = aim_pt;
    }
    return true;
}

// ---- 默认实现 ----

void PathFollower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    const auto& fk = ctx->config.control.fork;
    pathplan::handleFork(wide_mask, fork_state_,
                         fk.keep_side, fk.confirm_rows,
                         fk.persist_frames, fk.width_threshold, 1);
}

void PathFollower::onCenterline(const cv::Mat& mask,
                                 std::vector<cv::Point2f>& centers,
                                 SharedContext* /*ctx*/) {
    pathplan::extractCenterline(mask, centers);
    pathplan::smoothCenterline(centers);
}

bool PathFollower::onPurePursuit(const cv::Mat& mask,
                                  const std::vector<cv::Point2f>& centers,
                                  float& error,
                                  cv::Point2f& aim_pt,
                                  SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at(ctx->scene_manager.currentName());
    const auto& calib = ctx->config.calib;

    int w = mask.cols, h = mask.rows;
    float cx = w * 0.5f, cy = h * ctrl.car_cy_ratio;
    float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : sp.max_speed;
    float aim_dist = sp.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_dist, calib.ppm_x, calib.ppm_y, aim_pt))
        return false;

    error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                    ctrl.car_length, calib.ppm_x, calib.ppm_y)
          * ctx->steering_dir;
    return true;
}

void PathFollower::onLost(const cv::Point2f& e_pt, float yaw_rad,
                           SharedContext* ctx) {
    const auto& ctrl  = ctx->config.control;
    const auto& sp    = ctrl.scene.at(ctx->scene_manager.currentName());
    const auto& calib = ctx->config.calib;

    // 读当前车坐标
    cv::Point2f car_pt;
    {
        std::lock_guard<std::mutex> lk(ctx->guide_mtx_);
        car_pt = ctx->guide_meta_.car_pt;
    }

    // 纯跟踪锁冻结 E 点 + yaw 补偿
    float error = pathplan::headingError(car_pt.x, car_pt.y,
                                          e_pt.x, e_pt.y,
                                          ctrl.car_length, calib.ppm_x, calib.ppm_y)
                * ctx->steering_dir
                - yaw_rad;

    // PD + 速度（正常，不衰减）
    float derivative = first_frame_ ? 0.0f : (error - last_error_);
    last_error_ = error;

    float pwm = ctrl.pwm_mid
              - (sp.pid_p * error + sp.pid_d * derivative) * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
    last_pwm_ = pwm;

    float target_spd = sp.max_speed;
    float ae = std::abs(error);
    if (ae > ctrl.steer_speed_threshold) {
        float excess = ae - ctrl.steer_speed_threshold;
        target_spd = sp.max_speed / (1.0f + ctrl.steer_speed_k * excess);
    }
    if (target_spd < last_speed_) last_speed_ = target_spd;
    else last_speed_ += (target_spd - last_speed_) * sp.speed_recovery_k;

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(last_speed_);
    ctx->last_error.store(last_error_, std::memory_order_release);
}

float PathFollower::onPD(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp   = ctrl.scene.at(ctx->scene_manager.currentName());

    float derivative = first_frame_ ? 0.0f : (error - last_error_);
    last_error_ = error;

    float output = sp.pid_p * error + sp.pid_d * derivative;
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    return std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
}

float PathFollower::onSpeed(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp   = ctrl.scene.at(ctx->scene_manager.currentName());

    float target = sp.max_speed;
    float ae = std::abs(error);
    if (ae > ctrl.steer_speed_threshold) {
        float excess = ae - ctrl.steer_speed_threshold;
        target = sp.max_speed / (1.0f + ctrl.steer_speed_k * excess);
    }

    // 减速立即生效，加速平滑恢复
    if (target < last_speed_) {
        return target;
    }
    return last_speed_ + (target - last_speed_) * sp.speed_recovery_k;
}
