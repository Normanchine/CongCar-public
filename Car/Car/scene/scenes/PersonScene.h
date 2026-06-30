#pragma once

#include "../framework/IScene.h"

class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    void doTracking(SharedContext* ctx, float target_speed);
    void resetGateState();

    // 4点方向门控: 1=10, 2=30, 3=610, 4=630
    bool  waiting_for_1_ = false;
    bool  waiting_for_4_ = false;
    bool  dashing_       = false;
    float last_proj_x_   = -1.0f;

    // 巡线状态
    float last_error_  = 0.0f;
    int   lost_frames_ = 0;
    float last_pwm_    = 1500.0f;
    float last_speed_  = 0.0f;
    bool  first_frame_ = true;
};
