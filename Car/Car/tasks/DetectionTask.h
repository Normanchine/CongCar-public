#pragma once

#include "../thread/IThreadTask.h"

class DetectionTask : public IThreadTask {
public:
    explicit DetectionTask(SharedContext* ctx);
    ~DetectionTask() override;

    void run() override;

private:
    uint64_t frame_counter_ = 0;
    int det_lost_frames_ = 0;  // 连续丢检测帧计数
};
