#pragma once
// 巡线 + 纯跟踪 + PD：纯算法，从分割 mask 计算转向指令
// 不依赖 SharedContext

#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

namespace pathplan {

// ---- 岔路处理：检测到岔路时抹除分支，保留主干 ----
// 在宽鸟瞰mask上操作
// 扫描方向：从近到远（底部→顶部），逐行找连通段
// 出现2段→岔路，按keep_side参数决定保留侧（0=左，1=右），抹除对侧
inline void handleFork(cv::Mat& mask, int keep_side = 0, int min_seg_width = 3) {
    int w = mask.cols, h = mask.rows;
    if (h < 5) return;

    bool fork_found = false;

    for (int r = h - 1; r >= 0; --r) {
        uint8_t* row = mask.ptr<uint8_t>(r);

        // ---- 找出本行所有连通段（从左到右有序） ----
        std::vector<std::array<int, 2>> segs;
        for (int c = 0; c < w; ) {
            while (c < w && row[c] == 0) ++c;
            if (c >= w) break;
            int l = c;
            while (c < w && row[c] > 0) ++c;
            int r_end = c - 1;
            if (r_end - l + 1 >= min_seg_width)
                segs.push_back({l, r_end});
        }

        if (segs.empty()) {
            if (fork_found) break;   // 岔路区域结束
            continue;
        }

        if (segs.size() >= 2) {
            // 岔路：按keep_side保留，抹除对侧分支
            int keep_idx = (keep_side == 0) ? 0 : static_cast<int>(segs.size()) - 1;
            for (int i = 0; i < static_cast<int>(segs.size()); ++i) {
                if (i == keep_idx) continue;
                for (int c = segs[i][0]; c <= segs[i][1]; ++c)
                    row[c] = 0;
            }
            fork_found = true;
        }
        // 单段：不操作
    }
}

// ---- 逐行扫描中线（纵向赛道：从左/右找白色像素，取中点） ----
inline void extractCenterlineVertical(const cv::Mat& mask,
                                      std::vector<cv::Point2f>& out) {
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

        out.emplace_back((left + right) * 0.5f, static_cast<float>(r));
    }
}

// ---- 中线提取：逐行扫描左右边界取中点 ----
inline void extractCenterline(const cv::Mat& mask,
                              std::vector<cv::Point2f>& out) {
    extractCenterlineVertical(mask, out);
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
