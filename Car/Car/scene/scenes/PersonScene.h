#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

// 检测到person时触发的场景：4点方向门控 + 冲刺
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 巡线跟随（可设外部速度）
    PathFollower follower_;

    // ---- 4点方向门控: 1=10, 2=30, 3=610, 4=630 ----
    void resetGateState();

    bool  waiting_for_1_ = false;
    bool  waiting_for_4_ = false;
    bool  dashing_       = false;
    float last_proj_x_   = -1.0f;
};
