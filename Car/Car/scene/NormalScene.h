#pragma once

#include "IScene.h"

class NormalScene : public IScene {
public:
    NormalScene() = default;
    ~NormalScene() override = default;

    void run(SharedContext* ctx) override;

private:
    float last_error_ = 0.0f;
};
