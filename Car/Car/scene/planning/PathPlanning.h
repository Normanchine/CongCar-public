#pragma once
// 巡线 + 纯跟踪 + PD：纯算法，从分割 mask 计算转向指令
// 不依赖 SharedContext

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

namespace pathplan {

// ---- 岔路持久化状态 ----
// 场景切换时由 PathFollower 重置
struct ForkState {
    int   persist_remaining = 0;    // 持久化剩余帧数，0=检测模式
    float persist_stem_w   = 0.0f;  // 持久化的主干宽度（像素）
    int   persist_keep_side = 0;    // 持久化的保留侧
};

// ---- 岔路处理：等宽切除 ----
// 原理：赛道宽度固定，每个分支宽度 ≈ 主干宽度。
//       从保留侧边界量 stem_w 像素，超出部分切除。
//       粘连区和分离区用同一规则一刀切。
//
// 检测模式 (state.persist_remaining == 0):
//   双向扫描，连续 fork_confirm_rows 行 segs≥2 → 岔路确认
//   stem_w = 平均(确认行的保留侧段宽 + 主干侧单段行宽)
//   从岔路确认区向分支方向等宽切除 → 进入持久化模式
//
// 持久化模式 (state.persist_remaining > 0):
//   不扫描、不重算 stem_w，直接用保存的 stem_w
//   每行宽度 > stem_w * width_threshold → 切除
//   帧数递减，归零后回到检测模式
inline void handleFork(cv::Mat& mask, ForkState& state,
                       int keep_side, int fork_confirm_rows,
                       int persist_frames, float width_threshold,
                       int min_seg_width = 3) {
    int w = mask.cols, h = mask.rows;
    if (h < 5) return;

    // ---- 辅助：提取一行的连通段 ----
    auto findSegs = [&](int r) {
        std::vector<std::array<int, 2>> segs;
        const uint8_t* row = mask.ptr<uint8_t>(r);
        for (int c = 0; c < w; ) {
            while (c < w && row[c] == 0) ++c;
            if (c >= w) break;
            int l = c;
            while (c < w && row[c] > 0) ++c;
            int r_end = c - 1;
            if (r_end - l + 1 >= min_seg_width)
                segs.push_back({l, r_end});
        }
        return segs;
    };

    // ---- 辅助：对单行等宽切除 ----
    // 参考左边界，keep_side=0保留左(切右)，keep_side=1保留右(切左)
    auto cutRow = [&](int r, int stem_w, const std::vector<std::array<int, 2>>& segs) {
        uint8_t* row = mask.ptr<uint8_t>(r);
        int left = segs[0][0];  // 最左段左边界
        if (keep_side == 0) {
            // 保留左：从 left+stem_w 起清零
            int cut = left + stem_w;
            if (cut < w) std::memset(row + cut, 0, (w - cut) * sizeof(uint8_t));
        } else {
            // 保留右：从 left 到 left+stem_w 清零
            int cut = std::min(left + stem_w, w);
            std::memset(row + left, 0, (cut - left) * sizeof(uint8_t));
        }
    };

    // ======== 持久化模式 ========
    if (state.persist_remaining > 0) {
        state.persist_remaining--;
        int stem_w = static_cast<int>(std::round(state.persist_stem_w));
        if (stem_w < 2) return;

        float thr = state.persist_stem_w * width_threshold;
        for (int r = 0; r < h; ++r) {
            auto segs = findSegs(r);
            if (segs.empty()) continue;
            // 整行总跨度
            int row_w = segs.back()[1] - segs[0][0] + 1;
            if (row_w <= thr) continue;  // 宽度正常，不切
            cutRow(r, stem_w, segs);
        }
        return;
    }

    // ======== 检测模式：一趟扫描，边扫边记 ========
    // dir: -1=Bottom→Top, +1=Top→Bottom
    // 主干在扫描来向（dir的反方向），分叉在扫描去往方向
    auto detectPass = [&](int r_start, int dir) -> bool {
        std::vector<int> stem_widths;   // 主干侧单段行宽度
        std::vector<int> fork_widths;   // 分叉行保留侧段宽度
        int fork_count = 0;
        int fork_r_start = 0;
        const int stem_max = 20;        // 最多保留的主干行数

        for (int r = r_start; r >= 0 && r < h; r += dir) {
            auto segs = findSegs(r);
            if (segs.empty()) {
                fork_count = 0;
                fork_widths.clear();
                continue;
            }

            if (segs.size() == 1) {
                // 单段：记录到主干宽度缓冲
                stem_widths.push_back(segs[0][1] - segs[0][0] + 1);
                if (static_cast<int>(stem_widths.size()) > stem_max)
                    stem_widths.erase(stem_widths.begin());
                fork_count = 0;
                fork_widths.clear();
            } else {
                // 多段：分叉行
                if (fork_count == 0) fork_r_start = r;
                fork_count++;

                // 记录最左段宽度（锚点统一左边界，stem_w=切除/保留量）
                int kw = segs[0][1] - segs[0][0] + 1;
                fork_widths.push_back(kw);

                if (fork_count >= fork_confirm_rows) {
                    // ★ 岔路确认！
                    // stem_w = 分叉确认行保留侧段宽的平均
                    float sum = 0.0f;
                    int cnt = 0;
                    for (int fw : fork_widths) { sum += fw; cnt++; }
                    if (cnt == 0) return false;

                    float stem_w = sum / cnt;
                    int stem_wi = static_cast<int>(std::round(stem_w));

                    // 统一规则：从分叉点向外两个方向，每行左边界+stem_w切除，遇空行停
                    for (int rr = fork_r_start; rr >= 0 && rr < h; rr += dir) {
                        auto cut_segs = findSegs(rr);
                        if (cut_segs.empty()) break;
                        cutRow(rr, stem_wi, cut_segs);
                    }
                    for (int rr = fork_r_start - dir; rr >= 0 && rr < h; rr -= dir) {
                        auto cut_segs = findSegs(rr);
                        if (cut_segs.empty()) break;
                        cutRow(rr, stem_wi, cut_segs);
                    }

                    // 进入持久化模式
                    state.persist_remaining = persist_frames;
                    state.persist_stem_w = stem_w;
                    state.persist_keep_side = keep_side;
                    return true;
                }
            }
        }
        return false;
    };

    // ======== Pass 1: Bottom→Top（近→远） ========
    if (detectPass(h - 1, -1)) return;

    // ======== Pass 2: Top→Bottom（远→近） ========
    detectPass(0, +1);
}

// ---- 逐行扫描赛道边界：edge=-1 取左边界, +1 取右边界, 0 取中点 ----
// CarScene 沿边界巡线时直接取单侧边界点（不取中点）
inline void extractBoundary(const cv::Mat& mask, std::vector<cv::Point2f>& out, int edge) {
    int w = mask.cols, h = mask.rows;
    out.clear();
    out.reserve(h);

    for (int r = 0; r < h; ++r) {
        const uint8_t* row = mask.ptr<uint8_t>(r);

        int left = -1, right = -1;
        for (int c = 0; c < w; ++c) {
            if (row[c] > 0) { left = c; break; }
        }
        if (left < 0) continue;

        for (int c = w - 1; c >= left; --c) {
            if (row[c] > 0) { right = c; break; }
        }
        if (right < 0) continue;

        float x;
        if (edge < 0)      x = static_cast<float>(left);
        else if (edge > 0) x = static_cast<float>(right);
        else               x = (left + right) * 0.5f;
        out.emplace_back(x, static_cast<float>(r));
    }
}

// ---- 中线提取：逐行扫描左右边界取中点 ----
inline void extractCenterline(const cv::Mat& mask,
                              std::vector<cv::Point2f>& out) {
    extractBoundary(mask, out, 0);
}

// ---- 滑动平均滤波（只平滑中线偏移方向，保留原始扫描行/列坐标） ----
inline void smoothCenterline(std::vector<cv::Point2f>& centers, int window_size = 5) {
    if (centers.size() < static_cast<size_t>(window_size)) return;

    // 判断是横向还是纵向中线：X跨度 > Y跨度 → 横向
    bool is_horizontal = false;
    if (std::abs(centers[0].x - centers.back().x) >
        std::abs(centers[0].y - centers.back().y)) {
        is_horizontal = true;
    }

    std::vector<cv::Point2f> smoothed;
    smoothed.reserve(centers.size());
    int half_w = window_size / 2;

    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
        float sum = 0.0f;
        int count = 0;
        for (int j = std::max(0, i - half_w);
             j <= std::min(static_cast<int>(centers.size()) - 1, i + half_w); ++j) {
            sum += is_horizontal ? centers[j].y : centers[j].x;
            count++;
        }
        if (is_horizontal) {
            smoothed.emplace_back(centers[i].x, sum / count);  // 保留原始列号，平滑y
        } else {
            smoothed.emplace_back(sum / count, centers[i].y);  // 保留原始行号，平滑x
        }
    }
    centers = std::move(smoothed);
}

// ---- 纯跟踪：沿中线找距离车底 aim_dist_m 米的前视点 ----
// 起点为中线最近点（而非车底中心），弧长累加更准确
// ppm_x/ppm_y 分离转换，适应非正方形像素
inline bool findLookAhead(const std::vector<cv::Point2f>& centers,
                           float cx, float cy,
                           float aim_dist_m,
                           float ppm_x, float ppm_y,
                           cv::Point2f& out) {
    if (centers.empty()) return false;

    // 找到离车底中心最近的中线点作为弧长起点
    int best_i = 0;
    float best_d = 1e9f;
    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
        float dy = centers[i].y - cy;
        float dx = centers[i].x - cx;
        float d  = dx * dx + dy * dy;
        if (d < best_d) { best_d = d; best_i = i; }
    }

    // 从最近点沿中线累加距离（米）
    float accum = 0.0f;
    cv::Point2f prev = centers[best_i];
    for (int i = best_i - 1; i >= 0; --i) {
        float dy = (centers[i].y - prev.y) / ppm_y;
        float dx = (centers[i].x - prev.x) / ppm_x;
        accum += std::sqrt(dx * dx + dy * dy);
        if (accum >= aim_dist_m) {
            out = centers[i];
            return true;
        }
        prev = centers[i];
    }

    out = centers.front();
    return true;
}

// ---- 航向误差（纯跟踪模型） ----
// ppm_x: X方向像素/米，ppm_y: Y方向像素/米
inline float headingError(float cx, float cy, float ax, float ay,
                           float car_len, float ppm_x, float ppm_y) {
    // 将像素坐标转换为米
    float dx = (ax - cx) / ppm_x;           // X方向：像素 → 米
    float dy = (cy - ay) / ppm_y + car_len; // Y方向：像素 → 米 + 车长
    float dn = dx * dx + dy * dy;
    if (dn < 1e-6f) return 0.0f;
    return -std::atan(2.0f * car_len * dx / dn);
}

} // namespace pathplan
