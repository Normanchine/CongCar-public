#include "NormalScene.h"
#include "../../thread/SharedContext.h"
#include "../planning/PathPlanning.h"
#include "../framework/SceneFactory.h"
#include <algorithm>
#include <cmath>

REGISTER_SCENE(NormalScene)

void NormalScene::run(SharedContext* ctx) {
    //std::cout << "[NormalScene] 正在运行" << std::endl;
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // ---- 0. 岔路处理（透视变换前，在原始mask上抹除分支） ----
    cv::Mat mask_raw = seg->mask.clone();
    pathplan::handleFork(mask_raw);

    // ---- 1. 透视变换到鸟瞰 ----
    cv::Mat mask = ctx->perspective.warp(mask_raw);
    int w = mask.cols, h = mask.rows;
    float ppm_x = ctx->config.calib.ppm_x;
    float ppm_y = ctx->config.calib.ppm_y;

    // ---- 0.5 保存鸟瞰图mask（无论中线是否提取成功，始终保存供DisplayTask显示） ----
    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
        ctx->bird_mask_  = mask.clone();
        ctx->bird_valid_ = true;
    }

    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at("NormalScene");

    // ---- 1. 中线提取与平滑 ----
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(mask, centers);
    pathplan::smoothCenterline(centers);

    // ---- 2. 丢线容错：盲开 + 速度衰减 ----
    if (centers.empty()) {
        lost_frames_++;
        if (lost_frames_ > ctrl.max_lost_frames) {
            std::cerr << "[NormalScene] 赛道彻底丢失，执行停车保护" << std::endl;
            ctx->stop.store(true, std::memory_order_release);
        } else {
            std::cerr << "[NormalScene] 赛道闪烁丢失，盲开中... "
                      << lost_frames_ << "/" << ctrl.max_lost_frames << std::endl;
            // 沿丢线前方向逐步加大舵角，找回赛道
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

    // ---- 3. 车底中心 ----
    float cx = w * 0.5f;
    float cy = h * ctrl.car_cy_ratio;

    // ---- 4. 动态纯跟踪前视点 ----
    float current_speed = (last_speed_ > 0.01f) ? last_speed_ : sp.max_speed;
    float dynamic_aim = ctrl.aim_dist_near + current_speed * ctrl.lookahead_speed_k;
    cv::Point2f aim_pt;
    if (!pathplan::findLookAhead(centers, cx, cy, dynamic_aim, ppm_x, ppm_y, aim_pt)) return;

    // ---- 5. 航向误差（鸟瞰空间 ppm_x/ppm_y 分别转换） ----
    float error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                          ctrl.car_length, ppm_x, ppm_y)
                * ctx->steering_dir;

    // ---- 6. PD 控制（首帧跳过 D 项，防止场景切换脉冲） ----
    float derivative = 0.0f;
    if (!first_frame_) {
        derivative = error - last_error_;
    }
    last_error_ = error;

    float output = sp.pid_p * error + sp.pid_d * derivative;

    // ---- 7. PWM + 限幅 ----
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);

    last_pwm_ = pwm;

    // 转向幅度过大时降速（大弯减速）
    float target_speed = sp.max_speed;
    float abs_err = std::abs(error);
    if (abs_err > ctrl.steer_speed_threshold) {
        float excess = abs_err - ctrl.steer_speed_threshold;
        target_speed = sp.max_speed / (1.0f + 2.0f * excess);
    }

    // 加速缓冲：减速立即生效，加速（含首帧起步、弯后恢复、盲开恢复）一律平滑
    if (target_speed < last_speed_) {
        last_speed_ = target_speed;
    } else {
        last_speed_ += (target_speed - last_speed_) * ctrl.speed_recovery_k;
    }

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(last_speed_);

    first_frame_ = false;

    // ---- 8. 更新鸟瞰图中线/前视点（mask已在步骤0.5保存） ----
    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
        ctx->bird_centers_ = centers;
        ctx->bird_aim_pt_  = aim_pt;
    }
}
