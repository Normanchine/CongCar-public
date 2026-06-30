#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../planning/PathPlanning.h"
#include "../framework/SceneFactory.h"
#include <algorithm>
#include <cmath>

REGISTER_SCENE(PersonScene)

void PersonScene::run(SharedContext* ctx) {
    auto det = ctx->det_result.get();

    // 找最近的行人（class_id==9, y2最大）
    const DetectionBox* person = nullptr;
    if (det && !det->boxes.empty()) {
        float max_y2 = -1.0f;
        for (const auto& b : det->boxes) {
            if (b.class_id == 9 && b.y2 > max_y2) {
                max_y2 = b.y2;
                person = &b;
            }
        }
    }

    // 冲刺中人消失 → 继续冲刺
    if (dashing_ && !person) {
        doTracking(ctx, 1.5f);
        return;
    }

    // 没人且不在冲刺 → 等 DetectionTask 切回 NormalScene
    if (!person) return;

    float proj_x = (person->x1 + person->x2) * 0.5f;
    int   frame_h = ctx->frame_height.load(std::memory_order_acquire);
    float bottom_dist = static_cast<float>(frame_h) - person->y2;

    // ---- 阶段一：人还远（>5px），正常巡线 ----
    if (bottom_dist > 5.0f) {
        if (dashing_) {
            doTracking(ctx, 1.5f);
        } else {
            doTracking(ctx, ctx->config.control.scene.at("NormalScene").max_speed);
        }
        resetGateState();
        return;
    }

    // ---- 人已靠近（≤5px） ----

    // 冲刺中的人重新靠近 → 中止冲刺，回到等待
    if (dashing_) {
        dashing_ = false;
        resetGateState();
        last_proj_x_ = proj_x;
    }

    // ---- 阶段二：停车等待 + 方向门控 ----
    ctx->steering_pwm.store(ctx->config.control.pwm_mid);
    ctx->target_speed.store(0.0f);
    last_speed_ = 0.0f;

    if (last_proj_x_ > 0.0f) {
        // 左半区：点2(30)左向 → 点1(10)左向
        if (last_proj_x_ > 30.0f && proj_x <= 30.0f) {
            waiting_for_1_ = true;
            waiting_for_4_ = false;
        }
        if (waiting_for_1_ && last_proj_x_ > 10.0f && proj_x <= 10.0f) {
            dashing_ = true;
            first_frame_ = true;
        }

        // 右半区：点3(610)右向 → 点4(630)右向
        if (last_proj_x_ < 610.0f && proj_x >= 610.0f) {
            waiting_for_4_ = true;
            waiting_for_1_ = false;
        }
        if (waiting_for_4_ && last_proj_x_ < 630.0f && proj_x >= 630.0f) {
            dashing_ = true;
            first_frame_ = true;
        }

        // 反向跨越 → 掉头，清空对侧序列
        if (last_proj_x_ < 30.0f && proj_x >= 30.0f) waiting_for_1_ = false;
        if (last_proj_x_ > 610.0f && proj_x <= 610.0f) waiting_for_4_ = false;
    }
    last_proj_x_ = proj_x;
}

void PersonScene::doTracking(SharedContext* ctx, float target_speed) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // 宽鸟瞰变换 → 岔路处理 → 裁中心区域
    cv::Mat wide_mask = ctx->perspective.warp(seg->mask);

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->fork_mtx_);
        ctx->fork_raw_mask_ = wide_mask.clone();
    }

    pathplan::handleFork(wide_mask, ctx->config.control.fork_keep_side, 1);

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->fork_mtx_);
        ctx->fork_done_mask_ = wide_mask.clone();
        ctx->fork_valid_ = true;
    }

    int crop_w = ctx->config.calib.bird_width;
    int offset = (wide_mask.cols - crop_w) / 2;
    cv::Mat mask = wide_mask(cv::Rect(offset, 0, crop_w, wide_mask.rows)).clone();
    int w = mask.cols, h = mask.rows;
    float ppm_x = ctx->config.calib.ppm_x;
    float ppm_y = ctx->config.calib.ppm_y;

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
        ctx->bird_mask_  = mask.clone();
        ctx->bird_valid_ = true;
    }

    const auto& ctrl = ctx->config.control;
    const auto& sp   = ctrl.scene.at("NormalScene");

    // 中线提取与平滑
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(mask, centers);
    pathplan::smoothCenterline(centers);

    // 丢线容错
    if (centers.empty()) {
        lost_frames_++;
        if (lost_frames_ > ctrl.max_lost_frames) {
            std::cerr << "[PersonScene] 赛道彻底丢失，执行停车保护" << std::endl;
            ctx->stop.store(true, std::memory_order_release);
        } else {
            std::cerr << "[PersonScene] 赛道闪烁丢失，盲开中... "
                      << lost_frames_ << "/" << ctrl.max_lost_frames << std::endl;
            float sign = (last_error_ > 0.0f) ? 1.0f : -1.0f;
            float pwm = last_pwm_ - sign * ctrl.coast_steer_step;
            pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
            last_pwm_ = pwm;
            ctx->steering_pwm.store(pwm);
            last_speed_ = last_speed_ * ctrl.coast_speed_decay;
            ctx->target_speed.store(last_speed_);
        }
        return;
    }
    lost_frames_ = 0;

    // 车底中心
    float cx = w * 0.5f;
    float cy = h * ctrl.car_cy_ratio;

    // 动态纯跟踪前视点
    float current_speed = (last_speed_ > 0.01f) ? last_speed_ : target_speed;
    float dynamic_aim = ctrl.aim_dist_near + current_speed * ctrl.lookahead_speed_k;
    cv::Point2f aim_pt;
    if (!pathplan::findLookAhead(centers, cx, cy, dynamic_aim, ppm_x, ppm_y, aim_pt)) return;

    // 航向误差
    float error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                          ctrl.car_length, ppm_x, ppm_y)
                * ctx->steering_dir;

    // PD控制（首帧跳过D项）
    float derivative = 0.0f;
    if (!first_frame_) {
        derivative = error - last_error_;
    }
    last_error_ = error;

    float output = sp.pid_p * error + sp.pid_d * derivative;

    // PWM + 限幅
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
    last_pwm_ = pwm;

    // 大弯减速
    float final_speed = target_speed;
    float abs_err = std::abs(error);
    if (abs_err > ctrl.steer_speed_threshold) {
        float excess = abs_err - ctrl.steer_speed_threshold;
        final_speed = target_speed / (1.0f + 2.0f * excess);
    }

    // 减速立即生效，加速平滑恢复
    if (final_speed < last_speed_) {
        last_speed_ = final_speed;
    } else {
        last_speed_ += (final_speed - last_speed_) * ctrl.speed_recovery_k;
    }

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(last_speed_);

    first_frame_ = false;

    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
        ctx->bird_centers_ = centers;
        ctx->bird_aim_pt_  = aim_pt;
    }
}

void PersonScene::resetGateState() {
    waiting_for_1_ = false;
    waiting_for_4_ = false;
    dashing_       = false;
    last_proj_x_   = -1.0f;
}
