#include "NormalScene.h"
#include "../thread/SharedContext.h"
#include "PathPlanning.h"
#include "SceneFactory.h"
#include <algorithm>
#include <cmath>

REGISTER_SCENE(NormalScene)

void NormalScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // ---- 0. 透视变换到鸟瞰 ----
    cv::Mat mask = ctx->perspective.warp(seg->mask);
    int w = mask.cols, h = mask.rows;
    float ppm_x = ctx->config.calib.ppm_x;
    float ppm_y = ctx->config.calib.ppm_y;

    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at("NormalScene");

    // ---- 1. 中线提取与平滑 ----
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(mask, centers, ctrl.track_search_turn_dir);
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
            ctx->steering_pwm.store(last_pwm_);
            last_speed_ *= ctrl.coast_speed_decay;
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
                                          ctrl.car_length, ppm_x, ppm_y);

    // ---- 6. PD 控制（首帧跳过 D 项，防止场景切换脉冲） ----
    float derivative = 0.0f;
    if (first_frame_) {
        first_frame_ = false;
    } else {
        derivative = error - last_error_;
    }
    last_error_ = error;

    float output = sp.pid_p * error + sp.pid_d * derivative;

    // ---- 7. PWM + 限幅 ----
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);

    last_pwm_ = pwm;
    last_speed_ = sp.max_speed;

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(sp.max_speed);

    // ---- 8. 保存鸟瞰图中线数据（供 DisplayTask PIP 显示） ----
    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
        ctx->bird_mask_    = mask.clone();
        ctx->bird_centers_ = centers;
        ctx->bird_aim_pt_  = aim_pt;
        ctx->bird_valid_   = true;
    }
}
