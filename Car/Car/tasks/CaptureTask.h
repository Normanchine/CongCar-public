#pragma once

#include "../thread/IThreadTask.h"

class CaptureTask : public IThreadTask {
public:
    explicit CaptureTask(SharedContext* ctx);
    ~CaptureTask() override;

    void run() override;
};
