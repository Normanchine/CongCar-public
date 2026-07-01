#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

class NormalScene : public IScene {
public:
    NormalScene() = default;
    ~NormalScene() override = default;

    void run(SharedContext* ctx) override;

private:
    PathFollower follower_;
};
