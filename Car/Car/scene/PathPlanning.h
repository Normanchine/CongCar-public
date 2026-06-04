#pragma once
// 巡线 + 纯跟踪 + PD：纯算法，从分割 mask 计算转向指令
// 不依赖 SharedContext

#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

namespace pathplan {

// ---- 滑动平均滤波器（用于中线点平滑） ----
inline void movingAverageFilter(std::vector<cv::Point2f>& points, int window_size = 5) {
    if (points.size() < static_cast<size_t>(window_size) || window_size < 3) {
        return;  // 点数太少，不滤波
    }
    
    std::vector<cv::Point2f> filtered;
    filtered.reserve(points.size());
    
    int half_win = window_size / 2;
    for (size_t i = 0; i < points.size(); ++i) {
        float sum_x = 0.0f, sum_y = 0.0f;
        int count = 0;
        
        // 滑动窗口
        for (int j = std::max(0, static_cast<int>(i) - half_win);
             j <= std::min(static_cast<int>(points.size()) - 1, static_cast<int>(i) + half_win);
             ++j) {
            sum_x += points[j].x;
            sum_y += points[j].y;
            ++count;
        }
        
        filtered.emplace_back(sum_x / count, sum_y / count);
    }
    
    points = std::move(filtered);
}

// ---- 中值滤波器（去除野点） ----
inline void medianFilter(std::vector<cv::Point2f>& points, int window_size = 5) {
    if (points.size() < static_cast<size_t>(window_size) || window_size < 3) {
        return;
    }
    
    std::vector<cv::Point2f> filtered;
    filtered.reserve(points.size());
    
    int half_win = window_size / 2;
    for (size_t i = 0; i < points.size(); ++i) {
        std::vector<float> x_vals, y_vals;
        x_vals.reserve(window_size);
        y_vals.reserve(window_size);
        
        for (int j = std::max(0, static_cast<int>(i) - half_win);
             j <= std::min(static_cast<int>(points.size()) - 1, static_cast<int>(i) + half_win);
             ++j) {
            x_vals.push_back(points[j].x);
            y_vals.push_back(points[j].y);
        }
        
        // 排序取中值
        std::sort(x_vals.begin(), x_vals.end());
        std::sort(y_vals.begin(), y_vals.end());
        
        filtered.emplace_back(x_vals[x_vals.size() / 2], y_vals[y_vals.size() / 2]);
    }
    
    points = std::move(filtered);
}

// ---- 误差滑动平均（用于 PD 控制输入） ----
inline float smoothError(float new_error, float prev_smooth_error, float alpha = 0.3f) {
    // 一阶低通滤波：y[n] = α * x[n] + (1-α) * y[n-1]
    // alpha 越小，滤波效果越强，但延迟越大
    return alpha * new_error + (1.0f - alpha) * prev_smooth_error;
}

// ---- 误差滑动平均（窗口版） ----
inline float movingAverageError(const std::vector<float>& error_history) {
    if (error_history.empty()) return 0.0f;
    
    float sum = std::accumulate(error_history.begin(), error_history.end(), 0.0f);
    return sum / error_history.size();
}

// ---- 逐行扫描中线（从左/右找白色像素，取中点） ----
// 可选滤波参数：window_size=0 不滤波，>0 应用滤波
inline void extractCenterline(const cv::Mat& mask,
                               std::vector<cv::Point2f>& out,
                               int filter_window = 5,
                               bool use_median = false) {
    int w = mask.cols, h = mask.rows;
    out.clear();
    out.reserve(h);

    // ---- 1. 逐行扫描提取原始中线 ----
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
    
    // ---- 2. 可选：中值滤波去野点 ----
    if (use_median && filter_window >= 3) {
        medianFilter(out, filter_window);
    }
    // ---- 3. 可选：滑动平均平滑 ----
    else if (filter_window >= 3) {
        movingAverageFilter(out, filter_window);
    }
}

// ---- 纯跟踪：沿中线找距离车底 aim_dist_m 米的前视点 ----
// 注意：aim_dist_m 是实际距离（米），函数内部会根据 ppm_x/ppm_y 转换
inline bool findLookAhead(const std::vector<cv::Point2f>& centers,
                           float cx, float cy,
                           float aim_dist_m,
                           float ppm_x, float ppm_y,
                           cv::Point2f& out) {
    if (centers.empty()) return false;

    // 找到离车底中心最近的点作为起点
    int best_i = 0;
    float best_d = 1e9f;
    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
        float dy = (centers[i].y - cy) / ppm_y;  // 转为米
        float dx = (centers[i].x - cx) / ppm_x;  // 转为米
        float d  = dx * dx + dy * dy;
        if (d < best_d) { best_d = d; best_i = i; }
    }

    // 沿中线累加距离，直到达到前视距离
    float accum = 0.0f;
    cv::Point2f prev = {cx, cy};
    for (int i = best_i; i < static_cast<int>(centers.size()); ++i) {
        float dy = (centers[i].y - prev.y) / ppm_y;  // 转为米
        float dx = (centers[i].x - prev.x) / ppm_x;  // 转为米
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
    float dx = (ax - cx) / ppm_x;  // X方向：像素 → 米
    float dy = (cy - ay) / ppm_y + car_len;  // Y方向：像素 → 米 + 车长
    float dn = dx * dx + dy * dy;
    if (dn < 1e-6f) return 0.0f;
    return -std::atan(2.0f * car_len * dx / dn);
}

} // namespace pathplan
