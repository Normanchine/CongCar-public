#pragma once

#include "../thread/IThreadTask.h"

class SegmentationTask : public IThreadTask {
public:
    explicit SegmentationTask(SharedContext* ctx);
    ~SegmentationTask() override;

    void run() override;
};
