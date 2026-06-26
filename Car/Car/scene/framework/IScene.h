#pragma once

struct SharedContext;

class IScene {
public:
    virtual void run(SharedContext* ctx) = 0;
    virtual ~IScene() = default;
};
