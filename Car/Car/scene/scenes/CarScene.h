#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;

private:
    PathFollower follower_;
    int car_class_id_ = -1;   // 首次运行时从config查找
};
