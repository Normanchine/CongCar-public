#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

class RLightScene : public IScene {
public:
    RLightScene() = default;
    ~RLightScene() override = default;

    void run(SharedContext* ctx) override;

private:
    PathFollower follower_;
    int r_light_class_id_ = -1;  // 红灯类别ID（缓存）
};
