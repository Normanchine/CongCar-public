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
    float ppm = ctx->config.calib.ppm;

    // ---- 1. 中线提取 ----
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(mask, centers);
    if (centers.empty()) return;

    const auto& ctrl = ctx->config.control;

    // ---- 2. 车底中心 ----
    float cx = w * 0.5f;
    float cy = h * ctrl.car_cy_ratio;

    // ---- 3. 纯跟踪前视点 ----
    float aim_px = ctrl.aim_dist_near * ppm;
    cv::Point2f aim_pt;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_px, aim_pt)) return;

    // ---- 4. 航向误差（鸟瞰空间 ppm 全局一致） ----
    float error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                          ctrl.car_length, ppm);

    // ---- 5. PD 控制 ----
    float derivative = error - last_error_;
    last_error_ = error;

    float output = ctrl.pid_p * error + ctrl.pid_d * derivative;

    // ---- 6. PWM + 限幅 ----
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(ctrl.max_speed);
}
