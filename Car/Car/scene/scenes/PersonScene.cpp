#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <algorithm>
#include <cmath>

REGISTER_SCENE(PersonScene)

void PersonScene::run(SharedContext* ctx) {
    updatePersonClassId(ctx);
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    PersonTarget person = findClosestPerson(ctx);

    if (release_hold_frames_ > 0) {
        if (person.valid && person.distance_m <= sp.person_stop_dist_m) {
            float center_x = ctx->config.calib.bird_width * 0.5f;
            float center_band = ctx->config.calib.bird_width * sp.person_center_band_ratio;
            if (std::abs(person.bird_x - center_x) <= center_band) {
                startWaiting(person, ctx);
                return;
            }
        }
        release_hold_frames_--;
        followLine(ctx);
        return;
    }

    if (!person.valid) {
        if (waiting_) {
            lost_frames_++;
            if (lost_frames_ < sp.person_lost_release_frames) {
                stopCar(ctx);
                return;
            }
        }
        resetWaitState();
        followLine(ctx);
        return;
    }

    if (!waiting_ && person.distance_m > sp.person_stop_dist_m) {
        resetWaitState();
        followLine(ctx);
        return;
    }

    if (!waiting_) {
        startWaiting(person, ctx);
        return;
    }

    lost_frames_ = 0;
    stopCar(ctx);

    const float center_x = ctx->config.calib.bird_width * 0.5f;
    const float center_band = ctx->config.calib.bird_width * sp.person_center_band_ratio;
    const float delta_from_center = person.bird_x - center_x;
    const bool in_center_band = std::abs(delta_from_center) <= center_band;
    bool moving_in_center = false;

    if (last_bird_x_ >= 0.0f) {
        float dx = person.bird_x - last_bird_x_;
        moving_in_center = in_center_band && std::abs(dx) >= sp.person_move_px;
    }
    last_bird_x_ = person.bird_x;

    bool clear_by_distance = person.distance_m >= sp.person_release_dist_m;

    if (clear_by_distance || moving_in_center) {
        releaseAndFollow(ctx);
    }
}

void PersonScene::updatePersonClassId(SharedContext* ctx) {
    if (person_class_id_ >= 0) return;

    const auto& classes = ctx->config.ai.detection.classes;
    for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
        if (classes[i] == "person") {
            person_class_id_ = i;
            break;
        }
    }
}

PersonScene::PersonTarget PersonScene::findClosestPerson(SharedContext* ctx) const {
    PersonTarget best;
    if (person_class_id_ < 0) return best;

    auto det = ctx->det_result.get();
    if (!det || det->boxes.empty()) return best;

    const auto& ctrl = ctx->config.control;
    const auto& calib = ctx->config.calib;
    const int crop_off = (ctx->perspective.birdW() - calib.bird_width) / 2;
    const float car_y = calib.bird_height * ctrl.car_cy_ratio;

    for (const auto& b : det->boxes) {
        if (b.class_id != person_class_id_ || b.confidence < 0.3f) continue;

        cv::Point2f foot_img((b.x1 + b.x2) * 0.5f, b.y2);
        cv::Point2f foot_bird = ctx->perspective.toBird(foot_img);
        foot_bird.x -= static_cast<float>(crop_off);

        if (foot_bird.y < 0.0f || foot_bird.y > static_cast<float>(calib.bird_height)) {
            continue;
        }

        float dist_m = (car_y - foot_bird.y) / calib.ppm_y;
        if (dist_m < -0.05f) continue;

        if (!best.valid || dist_m < best.distance_m) {
            best.valid = true;
            best.bird_x = foot_bird.x;
            best.bird_y = foot_bird.y;
            best.distance_m = dist_m;
        }
    }

    return best;
}

void PersonScene::stopCar(SharedContext* ctx) {
    ctx->steering_pwm.store(ctx->config.control.pwm_mid);
    ctx->target_speed.store(0.0f);
}

void PersonScene::followLine(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (seg && !seg->mask.empty()) {
        follower_.process(seg->mask, ctx);
    }
}

void PersonScene::startWaiting(const PersonTarget& person, SharedContext* ctx) {
    waiting_ = true;
    lost_frames_ = 0;
    release_hold_frames_ = 0;
    last_bird_x_ = person.bird_x;

    stopCar(ctx);
}

void PersonScene::releaseAndFollow(SharedContext* ctx) {
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    resetWaitState();
    release_hold_frames_ = std::max(0, sp.person_release_hold_frames);
    ctx->target_speed.store(sp.max_speed);
    followLine(ctx);
}

void PersonScene::resetWaitState() {
    waiting_ = false;
    lost_frames_ = 0;
    last_bird_x_ = -1.0f;
}
