#pragma once

#include "../framework/IScene.h"

// 检测到person时触发的场景
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;
};
