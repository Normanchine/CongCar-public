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
};
