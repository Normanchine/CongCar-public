#pragma once

#include "../thread/IThreadTask.h"

class DisplayTask : public IThreadTask {
public:
    explicit DisplayTask(SharedContext* ctx);
    ~DisplayTask() override;

    void run() override;
};
