#pragma once
// 巡线 + 纯跟踪 + PD：纯算法，从分割 mask 计算转向指令
// 不依赖 SharedContext

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace pathplan {

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

// ---- 逐列扫描中线（横向赛道：从上/下找白色像素，取中点） ----
inline void extractCenterlineHorizontal(const cv::Mat& mask,
                                        std::vector<cv::Point2f>& out,
                                        int turn_dir) {
    int w = mask.cols, h = mask.rows;
    out.clear();
    out.reserve(w);

    // turn_dir == 1 (向右转): 从左向右构建，纯跟踪在右侧找前视点
    // turn_dir == -1 (向左转): 从右向左构建，纯跟踪在左侧找前视点
    int start_c = (turn_dir == 1) ? 0 : w - 1;
    int end_c   = (turn_dir == 1) ? w : -1;
    int step    = (turn_dir == 1) ? 1 : -1;

    for (int c = start_c; c != end_c; c += step) {
        int top = -1, bottom = -1;
        for (int r = 0; r < h; ++r) {
            if (mask.ptr<uint8_t>(r)[c] > 0) { top = r; break; }
        }
        if (top < 0) continue;

        for (int r = h - 1; r >= top; --r) {
            if (mask.ptr<uint8_t>(r)[c] > 0) { bottom = r; break; }
        }
        if (bottom < 0) continue;

        out.emplace_back(static_cast<float>(c), (top + bottom) * 0.5f);
    }
}

// ---- 综合中线提取：基于图像矩自动判断赛道走向 ----
inline void extractCenterline(const cv::Mat& mask,
                              std::vector<cv::Point2f>& out,
                              int turn_dir = 1) {
    cv::Moments m = cv::moments(mask, true);

    // 面积太小，返回空
    if (m.m00 < 100) {
        out.clear();
        return;
    }

    // 计算赛道主轴倾角
    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;

    // theta 范围 -pi/2 到 pi/2
    double theta = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02);
    double angle_deg = std::abs(theta * 180.0 / CV_PI);

    // 夹角 > 45° → 横向赛道（垂直发车）
    if (angle_deg > 45.0) {
        extractCenterlineHorizontal(mask, out, turn_dir);
    } else {
        extractCenterlineVertical(mask, out);
    }
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
    for (int i = best_i + 1; i < static_cast<int>(centers.size()); ++i) {
        float dy = (centers[i].y - prev.y) / ppm_y;
        float dx = (centers[i].x - prev.x) / ppm_x;
        accum += std::sqrt(dx * dx + dy * dy);
        if (accum >= aim_dist_m) {
            out = centers[i];
            return true;
        }
        prev = centers[i];
    }

    out = centers.back();
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
