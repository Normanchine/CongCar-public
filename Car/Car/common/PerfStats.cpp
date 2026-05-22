#include "PerfStats.h"
#include "logger.hpp"
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>

static Logger perfLogger("PerfStats");

PerfStats::PerfStats() {
    slots_[SLOT_CAPTURE].name      = "Capture";
    slots_[SLOT_SEGMENTATION].name = "Seg";
    slots_[SLOT_DETECTION].name    = "Detect";
    slots_[SLOT_CONTROL].name      = "Control";
    slots_[SLOT_DISPLAY].name      = "Display";
}

void PerfStats::record(int slot_id, int64_t elapsed_us) {
    auto& s = slots_[slot_id];
    s.frame_count.fetch_add(1, std::memory_order_release);
    s.total_time_us.fetch_add(elapsed_us, std::memory_order_release);
    s.last_frame_time_us.store(elapsed_us, std::memory_order_release);
}

void PerfStats::recordStages(int slot_id,
                              int64_t total_us, int64_t infer_us, int64_t post_us) {
    auto& s = slots_[slot_id];
    s.frame_count.fetch_add(1, std::memory_order_release);
    s.total_time_us.fetch_add(total_us, std::memory_order_release);
    s.infer_acc_us.fetch_add(infer_us, std::memory_order_release);
    s.post_acc_us.fetch_add(post_us, std::memory_order_release);
    s.last_frame_time_us.store(total_us, std::memory_order_release);
}

double PerfStats::getFps(int slot_id) const {
    auto t = slots_[slot_id].last_frame_time_us.load(std::memory_order_acquire);
    if (t > 0) return 1'000'000.0 / static_cast<double>(t);
    return 0.0;
}

void PerfStats::report() {
    // 计算真实报告间隔
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    int64_t prev_ns = last_report_ns_.load(std::memory_order_acquire);
    double interval_s = (prev_ns > 0) ? (now_ns - prev_ns) / 1e9 : 5.0;
    last_report_ns_.store(now_ns, std::memory_order_release);

    std::ostringstream header;
    header << "===== 性能统计 (" << std::fixed << std::setprecision(1)
           << interval_s << "s) =====";
    perfLogger.info(header.str());

    // 表头
    {
        std::ostringstream oss;
        oss << std::left  << std::setw(10) << "模块"
            << std::right << std::setw(8)  << "帧数"
            << std::setw(10) << "耗时ms"
            << std::setw(10) << "infer"
            << std::setw(10) << "post"
            << std::setw(10) << "FPS";
        perfLogger(oss.str());
    }

    for (int i = 0; i < SLOT_MAX; ++i) {
        auto cnt   = slots_[i].frame_count.exchange(0, std::memory_order_acquire);
        auto total = slots_[i].total_time_us.exchange(0, std::memory_order_acquire);
        auto infer = slots_[i].infer_acc_us.exchange(0, std::memory_order_acquire);
        auto post  = slots_[i].post_acc_us.exchange(0, std::memory_order_acquire);

        if (cnt == 0) continue;

        double avg_ms = (total > 0) ? total / 1000.0 / cnt : 0.0;
        double actual_fps = cnt / interval_s;

        std::ostringstream oss;
        oss << std::left  << std::setw(10) << slots_[i].name
            << std::right << std::setw(8)  << cnt
            << std::fixed << std::setprecision(2)
            << std::setw(10) << avg_ms;

        bool has_stages = (infer + post) > 0;
        if (has_stages) {
            double infer_avg = infer / 1000.0 / cnt;
            double post_avg  = post  / 1000.0 / cnt;
            oss << std::setw(10) << infer_avg
                << std::setw(10) << post_avg;
        } else {
            oss << std::setw(10) << "-"
                << std::setw(10) << "-";
        }

        oss << std::setprecision(1) << std::setw(10) << actual_fps;
        perfLogger(oss.str());
    }
}
