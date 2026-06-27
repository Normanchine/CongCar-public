#pragma once
// PP-YOLOE 后处理：anchor解码 + 坐标变换 + NMS
// 纯算法，对应 infer_rk3588.py 的 postprocess 逻辑
// 输入: 256×640, 中心裁剪一半高度

#include <algorithm>
#include <vector>

namespace ppyoloe {

// ---- 模型参数（与 onnx_io.json 一致） ----

constexpr int INPUT_H = 256;
constexpr int INPUT_W = 640;
constexpr int FPN_STRIDES[] = {32, 16, 8};
constexpr float GRID_OFFSET = 0.5f;
constexpr int NUM_CLASSES = 7;
constexpr int NUM_ANCHORS = 3360; // 8×20 + 16×40 + 32×80

// ---- 检测结果 ----

struct Detection {
    int class_id;
    float score;
    float x1, y1, x2, y2;
};

// ---- Anchor 网格（延迟初始化，线程安全） ----

inline void getAnchors(const float*& out_points, const float*& out_strides) {
    static float points[NUM_ANCHORS * 2];
    static float strides[NUM_ANCHORS];
    static bool initialized = false;

    if (!initialized) {
        int idx = 0;
        for (int s : FPN_STRIDES) {
            int fh = INPUT_H / s;
            int fw = INPUT_W / s;
            for (int i = 0; i < fh; ++i) {       // y 慢
                for (int j = 0; j < fw; ++j) {   // x 快
                    points[idx * 2 + 0] = static_cast<float>(j) + GRID_OFFSET;
                    points[idx * 2 + 1] = static_cast<float>(i) + GRID_OFFSET;
                    strides[idx] = static_cast<float>(s);
                    ++idx;
                }
            }
        }
        initialized = true;
    }
    out_points = points;
    out_strides = strides;
}

// ---- 主后处理 ----
// cls_data:  (12, 3360) float32, 已 sigmoid, row-major
// reg_data:  (3360, 4)  float32, ltrb 距离（grid cell 单位）, row-major
// orig_w, orig_h: 原始图片宽高
// crop_h:        中心裁剪后高度（= orig_h / 2）
// crop_top:      裁剪起始 Y 坐标（= (orig_h - crop_h) / 2）

inline std::vector<Detection> postprocess(
    const float* cls_data,
    const float* reg_data,
    int orig_w, int orig_h,
    int crop_h, int crop_top,
    float score_thresh = 0.3f,
    float nms_thresh = 0.65f)
{
    const float* anchor_pts;
    const float* anchor_strides;
    getAnchors(anchor_pts, anchor_strides);

    // 输入尺度 → 裁剪后坐标 的缩放因子
    const float scale_x = static_cast<float>(INPUT_W) / static_cast<float>(orig_w);
    const float scale_y = static_cast<float>(INPUT_H) / static_cast<float>(crop_h);

    std::vector<Detection> results;

    // 逐类处理
    for (int cls_id = 0; cls_id < NUM_CLASSES; ++cls_id) {
        // ---- 收集该类超过阈值的检测框 ----
        struct Candidate { float score; float x1, y1, x2, y2; };
        std::vector<Candidate> cands;
        cands.reserve(128);

        const float* cls_row = cls_data + cls_id * NUM_ANCHORS;

        for (int a = 0; a < NUM_ANCHORS; ++a) {
            float score = cls_row[a];
            if (score <= score_thresh) continue;

            // 解码 ltrb → x1y1x2y2 (grid cell 空间)
            float ax = anchor_pts[a * 2 + 0];
            float ay = anchor_pts[a * 2 + 1];
            float stride = anchor_strides[a];

            const float* reg = reg_data + a * 4;
            float l = reg[0], t = reg[1], r = reg[2], b = reg[3];

            float x1 = (ax - l) * stride;   // → 输入尺度 (256×640)
            float y1 = (ay - t) * stride;
            float x2 = (ax + r) * stride;
            float y2 = (ay + b) * stride;

            // 映射到裁剪后坐标
            x1 /= scale_x;
            x2 /= scale_x;
            y1 /= scale_y;
            y2 /= scale_y;

            // 恢复到原图坐标
            y1 += static_cast<float>(crop_top);
            y2 += static_cast<float>(crop_top);

            // 过滤无效框
            if (x2 <= x1 || y2 <= y1) continue;

            cands.push_back({score, x1, y1, x2, y2});
        }

        if (cands.empty()) continue;

        // ---- 按置信度降序排序 ----
        std::sort(cands.begin(), cands.end(),
            [](const Candidate& p, const Candidate& q) { return p.score > q.score; });

        // ---- 单类 NMS ----
        std::vector<bool> suppressed(cands.size(), false);
        for (size_t i = 0; i < cands.size(); ++i) {
            if (suppressed[i]) continue;

            const auto& ci = cands[i];
            float area_i = (ci.x2 - ci.x1) * (ci.y2 - ci.y1);

            // 钳位到图像边界 + 最终有效性检查（与Python一致：先钳位再检查）
            float cx1 = std::max(0.0f, ci.x1);
            float cy1 = std::max(0.0f, ci.y1);
            float cx2 = std::min(static_cast<float>(orig_w), ci.x2);
            float cy2 = std::min(static_cast<float>(orig_h), ci.y2);
            if (cx2 > cx1 && cy2 > cy1) {
                results.push_back({cls_id, ci.score, cx1, cy1, cx2, cy2});
            }

            // 抑制后续重叠框
            for (size_t j = i + 1; j < cands.size(); ++j) {
                if (suppressed[j]) continue;
                const auto& cj = cands[j];

                float iw = std::max(0.0f, std::min(ci.x2, cj.x2) - std::max(ci.x1, cj.x1) + 1e-5f);
                float ih = std::max(0.0f, std::min(ci.y2, cj.y2) - std::max(ci.y1, cj.y1) + 1e-5f);
                float inter = iw * ih;
                float area_j = (cj.x2 - cj.x1) * (cj.y2 - cj.y1);
                float iou = inter / (area_i + area_j - inter + 1e-6f);

                if (iou > nms_thresh) suppressed[j] = true;
            }
        }
    }

    // 按置信度降序排列
    std::sort(results.begin(), results.end(),
        [](const Detection& a, const Detection& b) { return a.score > b.score; });

    return results;
}

} // namespace ppyoloe
