#pragma once

#include <array>
#include <atomic>
#include <cstdint>

// 性能槽位索引
enum PerfSlotId : int {
    SLOT_CAPTURE      = 0,  // 采集线程
    SLOT_SEGMENTATION = 1,  // 语义分割线程
    SLOT_DETECTION    = 2,  // 目标检测线程
    SLOT_CONTROL      = 3,  // 控制线程
    SLOT_DISPLAY      = 4,  // 显示线程
    SLOT_MAX          = 5
};

struct PerfSlot {
    const char* name;
    std::atomic<uint64_t> frame_count{0};
    std::atomic<int64_t>  total_time_us{0};
    std::atomic<int64_t>  last_frame_time_us{0};
    std::atomic<int64_t>  pre_acc_us{0};    // pre阶段累计值(每报告周期重置)
    std::atomic<int64_t>  infer_acc_us{0};  // infer阶段累计值
    std::atomic<int64_t>  post_acc_us{0};   // post阶段累计值
};

class PerfStats {
public:
    PerfStats();

    // 各模块每处理完一帧调用，记录当帧总耗时(微秒)
    void record(int slot_id, int64_t elapsed_us);

    // AI模块专用：记录三个阶段耗时(微秒)
    void recordStages(int slot_id, int64_t pre_us, int64_t infer_us, int64_t post_us);

    // 获取指定槽位的瞬时帧率（基于最近一帧耗时），用于自适应调节
    double getFps(int slot_id) const;

    // T1定时器每5秒直接调用：打印报告并重置计数器
    void report();

    PerfStats(const PerfStats&) = delete;
    PerfStats& operator=(const PerfStats&) = delete;

private:
    std::array<PerfSlot, SLOT_MAX> slots_;
    std::atomic<int64_t> last_report_ns_{0};  // 上次报告的时间戳(ns)
};
