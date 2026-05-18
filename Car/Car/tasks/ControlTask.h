#pragma once

#include "../thread/IThreadTask.h"

class ControlTask : public IThreadTask {
public:
    explicit ControlTask(SharedContext* ctx);
    ~ControlTask() override;

    void run() override;
};
