#include "PathFollower.h"
#include "../../thread/SharedContext.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ---- 模板方法：固定流程 ----
bool PathFollower::process(const cv::Mat& seg_mask, SharedContext* ctx) {
    const auto& calib = ctx->config.calib;
    const auto& ctrl  = ctx->config.control;

    // 首帧从ctx继承error和speed，场景切换无脉冲
    if (first_frame_) {
        last_error_ = ctx->last_error.load(std::memory_order_acquire);
        last_speed_ = ctx->target_speed.load(std::memory_order_acquire);
    }

    // ---- 0. 宽鸟瞰变换 ----
    cv::Mat wide_mask = ctx->perspective.warp(seg_mask);

    // 岔路切除（handleFork）算法保留但不启用：沿边界巡线天然跟随边界，
    // 不再需要 mask 级岔路切除。如需启用，恢复 onFork(wide_mask, ctx) 调用。

    // ---- 1. 全程宽鸟瞰巡线（不再裁中心区，边界提取需要宽视野） ----
    cv::Mat mask = wide_mask;

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lk(ctx->bird_mtx_);
        ctx->bird_mask_  = mask.clone();
        ctx->bird_valid_ = true;
    }

    // ---- 3. 边界提取（可重写） ----
    std::vector<cv::Point2f> centers;
    onCenterline(mask, centers, ctx);

    // ---- 3.5 全局坐标系更新（供底部面板显示） ----
    {
        float birdW  = calib.bird_width * calib.bird_width_scale;
        float car_cx = birdW * 0.5f;
        float car_cy = static_cast<float>(calib.bird_height) * ctrl.car_cy_ratio;

        std::lock_guard<std::mutex> lk(ctx->guide_mtx_);
        auto& meta = ctx->guide_meta_;
        meta.origin  = {car_cx, car_cy};
        meta.car_pt  = {car_cx, car_cy};
        meta.car_yaw = 0.0f;

        if (!centers.empty()) {
            meta.e_pt  = {centers[0].x, centers[0].y};
            meta.blind = false;
        } else {
            meta.blind = true;
        }
    }

    // ---- 4. 丢线容错（可重写） ----
    if (centers.empty()) {
        onLost(ctx);
        return true;
    }
    lost_frames_ = 0;

    // ---- 5. 纯跟踪 + 航向误差（可重写） ----
    cv::Point2f aim_pt;
    float error;
    if (!onPurePursuit(mask, centers, error, aim_pt, ctx))
        return true;

    // 同步当前参考点偏移比例给显示（蓝点 = 控制器实际瞄准位置，含平滑值）
    ctx->ref_ratio.store(cur_cx_ratio_, std::memory_order_release);

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
    // 沿边巡线：默认取左边界（follow_edge_），CarScene 可切右边界
    pathplan::extractBoundary(mask, centers, follow_edge_);
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
    // 车底中心参考点目标：画面中心 - 沿边方向×横向偏移比例（贴边界内侧）
    // 左边界(-1)→偏右(cx增大)；右边界(+1)→偏左(cx减小)
    float target_cx_ratio = -follow_edge_ * sp.lateral_offset_ratio;

    // cx 平滑：NormalScene(cx_smooth_frames>0) 从画面中心/当前值线性平滑到目标；其他场景硬切
    if (sp.cx_smooth_frames > 0.0f && std::abs(target_cx_ratio - cur_cx_ratio_) > 1e-5f) {
        if (cx_smooth_left_ <= 0) cx_smooth_left_ = static_cast<int>(sp.cx_smooth_frames);
        cur_cx_ratio_ += (target_cx_ratio - cur_cx_ratio_) / cx_smooth_left_;
        cx_smooth_left_--;
    } else {
        cur_cx_ratio_ = target_cx_ratio;
    }

    float cx = w * (0.5f + cur_cx_ratio_);
    float cy = h * ctrl.car_cy_ratio;
    float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : ctrl.max_speed;
    // 平滑期间前视也用当前速度下的动态前视（最远），参考点移动时保持远前视稳定
    float aim_dist = ctrl.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
    // 前视点上限（场景专属，CarScene 限制前视避免远处误差放大；0=不限制）
    if (sp.aim_dist_max > 0.0f && aim_dist > sp.aim_dist_max)
        aim_dist = sp.aim_dist_max;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_dist, calib.ppm_x, calib.ppm_y, aim_pt))
        return false;

    error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                    ctrl.car_length, calib.ppm_x, calib.ppm_y)
          * ctx->steering_dir;
    return true;
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

    float step = ctrl.coast_steer_step * lost_frames_ / ctrl.max_lost_frames;
    // 盲开只右转：赛道通常在右侧，固定沿右转方向（pwm 增大）搜索找回赛道；
    // steering_dir 处理 AR 镜像，避免转向反
    float pwm = last_pwm_ + ctx->steering_dir * step;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
    last_pwm_ = pwm;
    ctx->steering_pwm.store(pwm);
    // 盲开固定速度：丢线后保持 coast_speed（不再指数衰减）
    last_speed_ = ctrl.coast_speed;
    ctx->target_speed.store(last_speed_);
}

float PathFollower::onPD(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;

    float derivative = first_frame_ ? 0.0f : (error - last_error_);
    last_error_ = error;

    float output = ctrl.pid_p * error + ctrl.pid_d * derivative;
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    return std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
}

float PathFollower::onSpeed(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;

    float target = ctrl.max_speed;
    float ae = std::abs(error);
    if (ae > ctrl.steer_speed_threshold) {
        float excess = ae - ctrl.steer_speed_threshold;
        target = ctrl.max_speed / (1.0f + ctrl.steer_speed_k * excess);
    }

    // 减速立即生效，加速平滑恢复
    if (target < last_speed_) {
        return target;
    }
    return last_speed_ + (target - last_speed_) * ctrl.speed_recovery_k;
}
