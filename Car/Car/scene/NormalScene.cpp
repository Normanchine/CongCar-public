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

    // ---- 1. 中线提取 ----
    std::vector<cv::Point2f> centers;
    pathplan::extractCenterline(mask, centers);
    if (centers.empty()) {
        std::cerr << "[NormalScene] 赛道丢失，退出程序" << std::endl;
        ctx->stop.store(true, std::memory_order_release);
        return;
    }

    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at("NormalScene");

    // ---- 2. 车底中心 ----
    float cx = w * 0.5f;
    float cy = h * ctrl.car_cy_ratio;

    // ---- 3. 纯跟踪前视点 ----
    cv::Point2f aim_pt;
    if (!pathplan::findLookAhead(centers, cx, cy, ctrl.aim_dist_near, ppm_x, ppm_y, aim_pt)) return;

    // ---- 4. 航向误差（鸟瞰空间 ppm_x/ppm_y 分别转换） ----
    float error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                          ctrl.car_length, ppm_x, ppm_y);

    // ---- 5. PD 控制 ----
    float derivative = error - last_error_;
    last_error_ = error;

    float output = sp.pid_p * error + sp.pid_d * derivative;

    // ---- 6. PWM + 限幅 ----
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    pwm = std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);

    ctx->steering_pwm.store(pwm);
    ctx->target_speed.store(sp.max_speed);

    // ---- 7. 保存鸟瞰图中线数据（供 DisplayTask PIP 显示） ----
    if (ctx->config.debug_mode) {
        std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
        ctx->bird_mask_    = mask.clone();
        ctx->bird_centers_ = centers;
        ctx->bird_aim_pt_  = aim_pt;
        ctx->bird_valid_   = true;
    }
}
