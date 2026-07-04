#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../../hal/uart/UartIO.h"
#include "../framework/SceneFactory.h"
#include <algorithm>
#include <cmath>

REGISTER_SCENE(PersonScene)

void PersonScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    const auto& sp = ctx->config.control.scene.at("PersonScene");
    cv::Point2f foot_img;
    bool person_detected = findPersonFoot(foot_img, ctx);

    // ---- 人消失 → 恢复行驶 ----
    if (!person_detected) {
        has_stopped_ = false;
        cross_count_ = 0;
        was_on_track_ = false;
        followLine(ctx);
        return;
    }

    // ---- 停车态：检查释放条件 ----
    if (has_stopped_) {
        bool on_track = isFootOnTrack(foot_img, seg->mask);

        // 上升沿：人从赛道外进入赛道 → 计一次穿越
        if (on_track && !was_on_track_) {
            cross_count_++;
            if (cross_count_ >= sp.person_cross_threshold) {
                // 释放：人已过中线
                has_stopped_ = false;
                cross_count_ = 0;
                was_on_track_ = false;
                followLine(ctx);
                return;
            }
        }
        was_on_track_ = on_track;

        // 不满足释放条件 → 继续停车
        stopCar(ctx);
        return;
    }

    // ---- 行驶态：检查是否需要停车 ----
    if (isPersonInDanger(foot_img, ctx)) {
        has_stopped_ = true;
        stopCar(ctx);
        return;
    }

    // 正常巡线
    followLine(ctx);
}

// ---- 查找最近行人（取 y2 最大的，即离车最近） ----
bool PersonScene::findPersonFoot(cv::Point2f& foot_img, SharedContext* ctx) const {
    if (person_class_id_ < 0) {
        // 首次查找 class_id
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "person") {
                const_cast<PersonScene*>(this)->person_class_id_ = i;
                break;
            }
        }
        if (person_class_id_ < 0) return false;
    }

    auto det = ctx->det_result.get();
    if (!det || det->boxes.empty()) return false;

    float best_y2 = -1.0f;
    bool found = false;

    for (const auto& b : det->boxes) {
        if (b.class_id != person_class_id_ || b.confidence < 0.3f) continue;
        if (b.y2 > best_y2) {
            best_y2 = b.y2;
            foot_img = cv::Point2f((b.x1 + b.x2) * 0.5f, b.y2);
            found = true;
        }
    }

    return found;
}

// ---- 纵向危险距离判断（鸟瞰投影） ----
bool PersonScene::isPersonInDanger(cv::Point2f foot_img, SharedContext* ctx) const {
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    const auto& calib = ctx->config.calib;
    const auto& ctrl = ctx->config.control;

    cv::Point2f foot_bird = ctx->perspective.toBird(foot_img);
    float car_y = calib.bird_height * ctrl.car_cy_ratio;
    float dist_m = (car_y - foot_bird.y) / calib.ppm_y;

    return dist_m >= 0.0f && dist_m <= sp.person_stop_dist_m;
}

// ---- 人脚底是否踩在赛道mask上（原图坐标系） ----
bool PersonScene::isFootOnTrack(cv::Point2f foot_img, const cv::Mat& seg_mask) const {
    int x = static_cast<int>(foot_img.x);
    int y = static_cast<int>(foot_img.y);

    if (x < 0 || x >= seg_mask.cols || y < 0 || y >= seg_mask.rows) return false;

    return seg_mask.at<uint8_t>(y, x) == 1;  // 赛道像素值 = 1
}

// ---- 停车（双保险：atomic + 串口直发） ----
void PersonScene::stopCar(SharedContext* ctx) {
    ctx->steering_pwm.store(ctx->config.control.pwm_mid);
    ctx->target_speed.store(0.0f);
    // 直接串口下发，不等ControlTask循环回来读atomic
    if (ctx->uart_io) {
        ctx->uart_io->sendControl(0.0f, static_cast<uint16_t>(ctx->config.control.pwm_mid));
    }
}

// ---- 巡线 ----
void PersonScene::followLine(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (seg && !seg->mask.empty()) {
        follower_.process(seg->mask, ctx);
    }
}

// ---- Follower::onSpeed：直接响应，不平滑加速 ----
float PersonScene::Follower::onSpeed(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp   = ctrl.scene.at("PersonScene");

    float target = sp.max_speed;
    float ae = std::abs(error);
    if (ae > ctrl.steer_speed_threshold) {
        float excess = ae - ctrl.steer_speed_threshold;
        target = sp.max_speed / (1.0f + ctrl.steer_speed_k * excess);
    }
    // 减速立即生效，加速也不平滑——直接返回目标值
    return target;
}
