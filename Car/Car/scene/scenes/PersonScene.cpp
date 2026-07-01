#include "PersonScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <algorithm>

//REGISTER_SCENE(PersonScene)

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
        auto seg = ctx->seg_result.get();
        if (seg && !seg->mask.empty())
            follower_.process(seg->mask, ctx);
        return;
    }

    // 没人且不在冲刺 → 继续循线，等 DetectionTask 切回 NormalScene
    if (!person) {
        auto seg = ctx->seg_result.get();
        if (seg && !seg->mask.empty())
            follower_.process(seg->mask, ctx);
        return;
    }

    float proj_x = (person->x1 + person->x2) * 0.5f;
    int   frame_h = ctx->frame_height.load(std::memory_order_acquire);
    float bottom_dist = static_cast<float>(frame_h) - person->y2;

    // ---- 阶段一：人还远（>5px），正常巡线 ----
    if (bottom_dist > 5.0f) {
        auto seg = ctx->seg_result.get();
        if (seg && !seg->mask.empty())
            follower_.process(seg->mask, ctx);
        resetGateState();
        return;
    }

    // ---- 人已靠近（≤5px） ----

    // 冲刺中的人重新靠近 → 中止冲刺
    if (dashing_) {
        dashing_ = false;
        resetGateState();
        last_proj_x_ = proj_x;
    }

    // ---- 阶段二：停车等待 + 方向门控 ----
    ctx->steering_pwm.store(ctx->config.control.pwm_mid);
    ctx->target_speed.store(0.0f);

    if (last_proj_x_ > 0.0f) {
        // 左半区：点2(30)左向 → 点1(10)左向
        if (last_proj_x_ > 30.0f && proj_x <= 30.0f) {
            waiting_for_1_ = true;
            waiting_for_4_ = false;
        }
        if (waiting_for_1_ && last_proj_x_ > 10.0f && proj_x <= 10.0f) {
            dashing_ = true;
        }

        // 右半区：点3(610)右向 → 点4(630)右向
        if (last_proj_x_ < 610.0f && proj_x >= 610.0f) {
            waiting_for_4_ = true;
            waiting_for_1_ = false;
        }
        if (waiting_for_4_ && last_proj_x_ < 630.0f && proj_x >= 630.0f) {
            dashing_ = true;
        }

        // 反向跨越 → 清空对侧序列
        if (last_proj_x_ < 30.0f && proj_x >= 30.0f) waiting_for_1_ = false;
        if (last_proj_x_ > 610.0f && proj_x <= 610.0f) waiting_for_4_ = false;
    }
    last_proj_x_ = proj_x;
}

void PersonScene::resetGateState() {
    waiting_for_1_ = false;
    waiting_for_4_ = false;
    dashing_       = false;
    last_proj_x_   = -1.0f;
}
