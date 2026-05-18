#pragma once
// 巡线 + 纯跟踪 + PD：纯算法，从分割 mask 计算转向指令
// 不依赖 SharedContext

#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

namespace pathplan {

// ---- 逐行扫描中线（从左/右找白色像素，取中点） ----
inline void extractCenterline(const cv::Mat& mask,
                               std::vector<cv::Point2f>& out) {
    int w = mask.cols, h = mask.rows;
    out.clear();
    out.reserve(h);

    for (int r = 0; r < h; ++r) {
        const uint8_t* row = mask.ptr<uint8_t>(r);

        int left = -1, right = -1;
        for (int c = 0; c < w; ++c) {
            if (row[c] == 1) { left = c; break; }
        }
        if (left < 0) continue;

        for (int c = w - 1; c >= left; --c) {
            if (row[c] == 1) { right = c; break; }
        }
        if (right < 0) continue;

        out.emplace_back((left + right) * 0.5f, static_cast<float>(r));
    }
}

// ---- 纯跟踪：沿中线找距离车底 aim_dist_px 像素的前视点 ----
inline bool findLookAhead(const std::vector<cv::Point2f>& centers,
                           float cx, float cy,
                           float aim_dist_px,
                           cv::Point2f& out) {
    if (centers.empty()) return false;

    int best_i = 0;
    float best_d = 1e9f;
    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
        float dy = centers[i].y - cy;
        float dx = centers[i].x - cx;
        float d  = dx * dx + dy * dy;
        if (d < best_d) { best_d = d; best_i = i; }
    }

    float accum = 0.0f;
    cv::Point2f prev = {cx, cy};
    for (int i = best_i; i < static_cast<int>(centers.size()); ++i) {
        float dy = centers[i].y - prev.y;
        float dx = centers[i].x - prev.x;
        accum += std::sqrt(dx * dx + dy * dy);
        if (accum >= aim_dist_px) {
            out = centers[i];
            return true;
        }
        prev = centers[i];
    }

    out = centers.back();
    return true;
}

// ---- 航向误差（纯跟踪模型） ----
inline float headingError(float cx, float cy, float ax, float ay,
                           float car_len, float ppm) {
    float dx = ax - cx;
    float dy = cy - ay + car_len * ppm;
    float dn = dx * dx + dy * dy;
    if (dn < 1e-6f) return 0.0f;
    return -std::atan(2.0f * car_len * ppm * dx / dn);
}

} // namespace pathplan
