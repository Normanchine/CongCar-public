#pragma once

#include "SharedContext.h"

class IThreadTask {
public:
    explicit IThreadTask(SharedContext* ctx) : ctx_(ctx) {}
    virtual ~IThreadTask() = default;

    // 线程主循环，由Car::start()在独立std::thread中调用
    virtual void run() = 0;

protected:
    SharedContext* ctx_;
};
