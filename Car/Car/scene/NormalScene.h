#pragma once

#include "IScene.h"

class NormalScene : public IScene {
public:
    NormalScene() = default;
    ~NormalScene() override = default;

    void run(SharedContext* ctx) override;

private:
    float last_error_  = 0.0f;
    int   lost_frames_ = 0;
    float last_pwm_    = 1500.0f;
    float last_speed_  = 0.0f;
    bool  first_frame_ = true;   // 首帧跳过D项，防止场景切换脉冲
};
