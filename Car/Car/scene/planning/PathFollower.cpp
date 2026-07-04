#include "PathFollower.h"
#include "../../thread/SharedContext.h"
#include <algorithm>
#include <cmath>

// ---- 模板方法：固定流程 ----
bool PathFollower::process(const cv::Mat& seg_mask, SharedContext* ctx) {
    // 首帧从ctx继承error和speed，场景切换无脉冲
    if (first_frame_) {
        last_error_ = ctx->last_error.load(std::memory_order_acquire);
        last_speed_ = ctx->target_speed.load(std::memory_order_acquire);
    }

    const auto& ctrl = ctx->config.control;
    const auto& scene_name = ctx->scene_manager.currentName();
    const auto& sp   = ctrl.scene.at(scene_name);
    const auto& calib = ctx->config.calib;

    // ---- 0. 宽鸟瞰变换 ----
    cv::Mat wide_mask = ctx->perspective.warp(seg_mask);

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->fork_mtx_);
        ctx->fork_raw_mask_ = wide_mask.clone();
    }

    // ---- 1. 岔路处理（可重写） ----
    onFork(wide_mask, ctx);

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->fork_mtx_);
        ctx->fork_done_mask_ = wide_mask.clone();
        ctx->fork_valid_ = true;
    }

    // ---- 2. 裁中心区域供路径规划 ----
    int crop_w = calib.bird_width;
    int offset = (wide_mask.cols - crop_w) / 2;
    cv::Mat mask = wide_mask(cv::Rect(offset, 0, crop_w, wide_mask.rows)).clone();
    float ppm_x = calib.ppm_x;
    float ppm_y = calib.ppm_y;

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->bird_mtx_);
        ctx->bird_mask_  = mask.clone();
        ctx->bird_valid_ = true;
    }

    // ---- 3. 中线提取（可重写） ----
    std::vector<cv::Point2f> centers;
    onCenterline(mask, centers, ctx);

    // ---- 4. 丢线容错（可重写） ----
    if (centers.empty()) {
        onLost(ctx);
        return true;
    }
    lost_frames_ = 0;

    // ---- 5. 纯跟踪 + 航向误差 ----
    int w = mask.cols, h = mask.rows;
    float cx = w * 0.5f, cy = h * ctrl.car_cy_ratio;
    if (scene_name == "CarScene") {
        float offset_ratio = std::clamp(sp.car_center_x_offset_ratio, -0.45f, 0.45f);
        cx += w * offset_ratio;
    }
    float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : sp.max_speed;
    float aim_dist = ctrl.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
    cv::Point2f aim_pt;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_dist, ppm_x, ppm_y, aim_pt))
        return true;

    float error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                         ctrl.car_length, ppm_x, ppm_y)
                * ctx->steering_dir;

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

void PathFollower::onLost(SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    lost_frames_++;

    if (lost_frames_ > ctrl.max_lost_frames) {
        std::cerr << "[PathFollower] 赛道彻底丢失，执行停车保护" << std::endl;
        ctx->stop.store(true, std::memory_order_release);
        return;
    }

    std::cerr << "[PathFollower] 赛道闪烁丢失，盲开中... "
              << lost_frames_ << "/" << ctrl.max_lost_frames << std::endl;

    float sign = (last_error_ > 0.0f) ? 1.0f : -1.0f;
    float step = ctrl.coast_steer_step * lost_frames_ / ctrl.max_lost_frames;
    float pwm = last_pwm_ - sign * step;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
    last_pwm_ = pwm;
    ctx->steering_pwm.store(pwm);
    last_speed_ *= ctrl.coast_speed_decay;
    ctx->target_speed.store(last_speed_);
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
    return last_speed_ + (target - last_speed_) * ctrl.speed_recovery_k;
}
