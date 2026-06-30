#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    PathFollower follower_;
};
