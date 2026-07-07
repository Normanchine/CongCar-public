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
    float crop_y = ctx->config.frame_crop.enable
        ? static_cast<float>(ctx->config.frame_crop.y) : 0.0f;
    cv::Point2f foot_img;
    bool person_detected = findPersonFoot(foot_img, ctx);
    bool hidden_car_person_risk =
        ctx->right_car_hidden_risk_frames.load(std::memory_order_acquire) > 0;

    // ---- 发车保持期：避免刚启动又因同一个人还可见而反复停车 ----
    if (release_hold_frames_ > 0) {
        release_hold_frames_--;
        if (release_hold_frames_ == 0) {
            clearCarPersonContext();
        }
        followLine(ctx);
        return;
    }

    // ---- 人短暂丢失：停车态先防抖，连续丢失后再恢复 ----
    if (!person_detected) {
        if (car_person_waiting_) {
            car_watch_frames_++;
            if (hidden_car_person_risk &&
                car_watch_frames_ < std::max(1, sp.person_car_watch_frames)) {
                stopCar(ctx);
                return;
            }
        } else if (waiting_) {
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

    lost_frames_ = 0;
    cv::Point2f foot_bird = ctx->perspective.toBird({foot_img.x, foot_img.y - crop_y});
    updatePersonTracker(foot_bird);

    // ---- 车人组合：右侧车近距离消失后先停死，只等人往右让出左偏走廊 ----
    if (hidden_car_person_risk || car_person_waiting_) {
        car_person_waiting_ = true;
        waiting_ = false;
        applyCarPersonContext(ctx);
        float avoid_offset_ratio = 0.0f;
        if (shouldReleaseCarPerson(foot_bird, ctx, avoid_offset_ratio)) {
            release_ready_frames_++;
            if (release_ready_frames_ >= std::max(1, sp.person_cross_threshold)) {
                releaseAndFollow(avoid_offset_ratio, ctx);
                return;
            }
        } else {
            release_ready_frames_ = 0;
        }

        stopCar(ctx);
        return;
    }

    // ---- 停车态：预测人离开通行走廊的提前发车窗口 ----
    if (waiting_) {
        float avoid_offset_ratio = 0.0f;
        if (shouldRelease(foot_bird, ctx, avoid_offset_ratio)) {
            release_ready_frames_++;
            if (release_ready_frames_ >= std::max(1, sp.person_cross_threshold)) {
                releaseAndFollow(avoid_offset_ratio, ctx);
                return;
            }
        } else {
            release_ready_frames_ = 0;
        }

        stopCar(ctx);
        return;
    }

    // ---- 行驶态：人进入车未来通行走廊才停车 ----
    if (isPersonInDanger(foot_bird, ctx)) {
        waiting_ = true;
        release_ready_frames_ = 0;
        stopCar(ctx);
        return;
    }

    // 正常巡线
    followLine(ctx);
}

// ---- 查找最近行人（取 y2 最大的，即离车最近） ----
bool PersonScene::findPersonFoot(cv::Point2f& foot_img, SharedContext* ctx) {
    if (person_class_id_ < 0) {
        // 首次查找 class_id
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "person") {
                person_class_id_ = i;
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

void PersonScene::updatePersonTracker(const cv::Point2f& foot_bird) {
    if (!tracker_valid_) {
        last_foot_bird_ = foot_bird;
        foot_dx_px_ = 0.0f;
        tracker_valid_ = true;
        return;
    }

    float dx = foot_bird.x - last_foot_bird_.x;
    foot_dx_px_ = foot_dx_px_ * 0.6f + dx * 0.4f;
    last_foot_bird_ = foot_bird;
}

// ---- 纵向距离 + 停车走廊判断（鸟瞰投影） ----
bool PersonScene::isPersonInDanger(const cv::Point2f& foot_bird, SharedContext* ctx) const {
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    const auto& calib = ctx->config.calib;
    const auto& ctrl = ctx->config.control;

    float car_y = calib.bird_height * ctrl.car_cy_ratio;
    float dist_m = (car_y - foot_bird.y) / calib.ppm_y;
    if (dist_m < 0.0f || dist_m > sp.person_stop_dist_m) return false;

    float center_x = ctx->perspective.birdW() * 0.5f;
    float half_width_px = std::max(1.0f, sp.person_stop_width_m * calib.ppm_x);
    return std::abs(foot_bird.x - center_x) <= half_width_px;
}

bool PersonScene::shouldRelease(const cv::Point2f& foot_bird,
                                SharedContext* ctx,
                                float& avoid_offset_ratio) {
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    const auto& calib = ctx->config.calib;

    avoid_offset_ratio = 0.0f;
    if (!tracker_valid_) return false;

    float abs_dx = std::abs(foot_dx_px_);
    if (abs_dx < sp.person_move_px) return false;

    float center_x = ctx->perspective.birdW() * 0.5f;
    float half_width_px = std::max(1.0f, sp.person_safe_width_m * calib.ppm_x);
    float left_edge = center_x - half_width_px;
    float right_edge = center_x + half_width_px;
    bool moving_right = foot_dx_px_ > 0.0f;

    float px_to_clear = moving_right ? (right_edge - foot_bird.x)
                                     : (foot_bird.x - left_edge);
    px_to_clear = std::max(0.0f, px_to_clear);
    float frames_to_clear = px_to_clear / abs_dx;
    if (frames_to_clear > static_cast<float>(sp.person_launch_lead_frames)) {
        return false;
    }

    float offset = moving_right ? -sp.person_avoid_offset_ratio
                                :  sp.person_avoid_offset_ratio;
    if (sp.person_disable_right_avoid_when_car && offset > 0.0f &&
        isRightSideCarPresent(ctx)) {
        offset = 0.0f;
    }

    avoid_offset_ratio = std::clamp(offset, -0.20f, 0.20f);
    return true;
}

bool PersonScene::shouldReleaseCarPerson(const cv::Point2f& foot_bird,
                                         SharedContext* ctx,
                                         float& avoid_offset_ratio) {
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    const auto& calib = ctx->config.calib;

    avoid_offset_ratio = 0.0f;
    if (!tracker_valid_ || foot_dx_px_ < sp.person_move_px) return false;

    float wide_center_x = ctx->perspective.birdW() * 0.5f;
    float left_offset = -std::abs(sp.person_car_left_avoid_offset_ratio);
    float corridor_center_x = wide_center_x + calib.bird_width * left_offset;
    float half_width_px = std::max(1.0f, sp.person_safe_width_m * calib.ppm_x);
    float right_edge = corridor_center_x + half_width_px;

    float px_to_clear = std::max(0.0f, right_edge - foot_bird.x);
    float frames_to_clear = px_to_clear / std::max(1.0f, foot_dx_px_);
    if (frames_to_clear > static_cast<float>(sp.person_car_launch_lead_frames)) {
        return false;
    }

    avoid_offset_ratio = std::clamp(left_offset, -0.20f, 0.0f);
    return true;
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

void PersonScene::resetWaitState() {
    waiting_ = false;
    car_person_waiting_ = false;
    lost_frames_ = 0;
    car_watch_frames_ = 0;
    release_ready_frames_ = 0;
    tracker_valid_ = false;
    foot_dx_px_ = 0.0f;
    clearCarPersonContext();
}

void PersonScene::releaseAndFollow(float avoid_offset_ratio, SharedContext* ctx) {
    const auto& sp = ctx->config.control.scene.at("PersonScene");
    waiting_ = false;
    car_person_waiting_ = false;
    lost_frames_ = 0;
    car_watch_frames_ = 0;
    release_ready_frames_ = 0;
    tracker_valid_ = false;
    foot_dx_px_ = 0.0f;
    release_hold_frames_ = std::max(0, sp.person_release_hold_frames);
    follower_.person_offset_ratio = avoid_offset_ratio;
    ctx->right_car_hidden_risk_frames.store(0, std::memory_order_release);
    ctx->target_speed.store(sp.max_speed);
    followLine(ctx);
}

void PersonScene::applyCarPersonContext(SharedContext* ctx) {
    const auto& car_sp = ctx->config.control.scene.at("CarScene");
    follower_.cut_side = 1; // 车后人场景只处理右侧出租车，继续切右
    follower_.car_context_offset_ratio =
        std::clamp(car_sp.car_center_x_offset_ratio, -0.45f, 0.45f);
}

void PersonScene::clearCarPersonContext() {
    follower_.cut_side = -1;
    follower_.car_context_offset_ratio = 0.0f;
    follower_.person_offset_ratio = 0.0f;
}

bool PersonScene::isRightSideCarPresent(SharedContext* ctx) {
    float dist_m = 0.0f;
    return findRightSideCar(dist_m, ctx);
}

bool PersonScene::findRightSideCar(float& dist_m, SharedContext* ctx) {
    if (car_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "car") {
                car_class_id_ = i;
                break;
            }
        }
        if (car_class_id_ < 0) return false;
    }

    auto det = ctx->det_result.get();
    if (!det || det->boxes.empty()) return false;

    const auto& ctrl = ctx->config.control;
    const auto& calib = ctx->config.calib;
    const auto& car_sp = ctx->config.control.scene.at("CarScene");
    float mid_x = ctx->perspective.birdW() * car_sp.car_mid_ratio;
    float car_y = calib.bird_height * ctrl.car_cy_ratio;
    float crop_y = ctx->config.frame_crop.enable
        ? static_cast<float>(ctx->config.frame_crop.y) : 0.0f;
    bool found = false;
    float best_dist = 1e9f;

    for (const auto& b : det->boxes) {
        if (b.class_id != car_class_id_ || b.confidence < 0.4f) continue;
        cv::Point2f bl = ctx->perspective.toBird({b.x1, b.y2 - crop_y});
        cv::Point2f br = ctx->perspective.toBird({b.x2, b.y2 - crop_y});
        float car_cx = (bl.x + br.x) * 0.5f;
        if (car_cx < mid_x) continue;

        float car_bottom_y = (bl.y + br.y) * 0.5f;
        float d = (car_y - car_bottom_y) / calib.ppm_y;
        if (!found || std::abs(d) < std::abs(best_dist)) {
            best_dist = d;
            found = true;
        }
    }

    if (!found) return false;
    dist_m = best_dist;
    return true;
}

void PersonScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    if (cut_side >= 0) {
        const auto& sp = ctx->config.control.scene.at("CarScene");
        float ratio = sp.car_cut_ratio;
        int w = wide_mask.cols;
        int cut_w = static_cast<int>(w * ratio);

        if (cut_side == 0) {
            wide_mask(cv::Rect(0, 0, cut_w, wide_mask.rows)) = 0;
        } else {
            wide_mask(cv::Rect(w - cut_w, 0, cut_w, wide_mask.rows)) = 0;
        }
    }

    PathFollower::onFork(wide_mask, ctx);
}

bool PersonScene::Follower::onPurePursuit(const cv::Mat& mask,
                                           const std::vector<cv::Point2f>& centers,
                                           float& error,
                                           cv::Point2f& aim_pt,
                                           SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at("PersonScene");
    const auto& calib = ctx->config.calib;

    int w = mask.cols, h = mask.rows;
    float cx = w * 0.5f;
    float offset_ratio = car_context_offset_ratio + person_offset_ratio;
    cx += w * std::clamp(offset_ratio, -0.45f, 0.45f);
    float cy = h * ctrl.car_cy_ratio;

    float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : sp.max_speed;
    float aim_dist = ctrl.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_dist, calib.ppm_x, calib.ppm_y, aim_pt))
        return false;

    error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                    ctrl.car_length, calib.ppm_x, calib.ppm_y)
          * ctx->steering_dir;
    return true;
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
