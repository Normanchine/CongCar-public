#pragma once

#include "IScene.h"

// 检测到car时触发的场景
class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;
};
