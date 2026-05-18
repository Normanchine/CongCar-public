#pragma once

#include "../thread/IThreadTask.h"

class DetectionTask : public IThreadTask {
public:
    explicit DetectionTask(SharedContext* ctx);
    ~DetectionTask() override;

    void run() override;

private:
    uint64_t frame_counter_ = 0;
};
