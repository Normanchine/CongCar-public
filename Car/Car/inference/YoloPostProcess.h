#pragma once
// YOLOv8 后处理：DFL解码 + 坐标变换 + 置信度过滤 + NMS
// 纯算法，不依赖 SharedContext / 线程

#include "../common/SharedResult.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define YOLO_HAS_NEON 1
#else
#define YOLO_HAS_NEON 0
#endif

namespace yolo {

constexpr int NUM_BRANCHES = 3;

const char* const CLASSES[] = {
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic_light","fire_hydrant","stop_sign","parking_meter","bench","bird",
    "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports_ball","kite","baseball_bat","baseball_glove","skateboard","surfboard",
    "tennis_racket","bottle","wine_glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot_dog","pizza",
    "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell_phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy_bear","hair_drier","toothbrush"
};
constexpr int NUM_CLASSES = 80;

struct RawBox { float x1, y1, x2, y2; float score; int cls; };

// ---- 预分配缓冲区：避免热路径堆分配 ----
struct PostProcessBuffers {
    static constexpr int TILE = 64;
    std::vector<float> tile_buf;   // TILE * max_channels

    void ensure(int max_channels) {
        if (static_cast<int>(tile_buf.size()) < TILE * max_channels)
            tile_buf.resize(TILE * max_channels);
    }
};

// ---- FastExp: Schraudolph 近似，误差 <2%，比 std::exp 快 5-8x ----
inline float fast_exp(float x) {
    x = std::max(-87.0f, std::min(87.0f, x));
    union { float f; int32_t i; } u;
    u.i = static_cast<int32_t>(x * 12102203.2f + 1064866805.0f);
    return u.f;
}

#if YOLO_HAS_NEON
// ---- NEON FastExp：4 路并行 Schraudolph ----
inline float32x4_t fast_exp_neon(float32x4_t x) {
    x = vmaxq_f32(vdupq_n_f32(-87.0f), vminq_f32(vdupq_n_f32(87.0f), x));
    float32x4_t scaled = vmlaq_f32(vdupq_n_f32(1064866805.0f), x, vdupq_n_f32(12102203.2f));
    int32x4_t i = vcvtq_s32_f32(scaled);
    return vreinterpretq_f32_s32(i);
}
#endif

// ---- 单像素标量 DFL 解码（余量回退） ----
inline void dfl_decode_one_scalar(const float* tile_buf, int mc, float* out) {
    constexpr int STRIDE = PostProcessBuffers::TILE;
    for (int p = 0; p < 4; ++p) {
        int ch0 = p * mc;
        float mv = tile_buf[ch0 * STRIDE];
        for (int m = 1; m < mc; ++m)
            if (tile_buf[(ch0 + m) * STRIDE] > mv)
                mv = tile_buf[(ch0 + m) * STRIDE];
        float sum = 0.0f, wgt = 0.0f;
        for (int m = 0; m < mc; ++m) {
            float ev = fast_exp(tile_buf[(ch0 + m) * STRIDE] - mv);
            sum += ev;
            wgt += ev * static_cast<float>(m);
        }
        out[p] = wgt / sum;
    }
}

// ---- 融合 DFL + box_process + filter_boxes（NEON 加速） ----
inline std::vector<RawBox> decode_branch(
    const float* box_data, const float* cls_data,
    int h, int w, int mc, int num_cls,
    int img_h, int img_w, float conf_thresh,
    PostProcessBuffers& buf)
{
    std::vector<RawBox> result;
    const int plane = h * w;
    const int channels = 4 * mc;
    constexpr int TILE = PostProcessBuffers::TILE;

    buf.ensure(channels);
    result.reserve(plane / 8);

    const float sx = static_cast<float>(img_w) / w;
    const float sy = static_cast<float>(img_h) / h;

    for (int tile_start = 0; tile_start < plane; tile_start += TILE) {
        const int tile_n = std::min(TILE, plane - tile_start);

        // 拷贝 box 通道到 tile 缓冲（消除跨 channel stride cache miss）
        for (int ch = 0; ch < channels; ++ch)
            std::memcpy(buf.tile_buf.data() + ch * TILE,
                        box_data + ch * plane + tile_start,
                        tile_n * sizeof(float));

        // 逐 tile 做 class argmax
        float  ts[TILE];
        int    tc[TILE];
        for (int i = 0; i < tile_n; ++i) {
            ts[i] = cls_data[tile_start + i];
            tc[i] = 0;
        }
        for (int c = 1; c < num_cls; ++c) {
            const float* src = cls_data + c * plane + tile_start;
            for (int i = 0; i < tile_n; ++i) {
                if (src[i] > ts[i]) { ts[i] = src[i]; tc[i] = c; }
            }
        }

        // DFL + 坐标变换 + 置信度过滤
#if YOLO_HAS_NEON
        int i = 0;
        for (; i + 3 < tile_n; i += 4) {
            const int px = tile_start + i;
            float dfl_buf[16]; // [side * 4 + pixel]

            for (int p = 0; p < 4; ++p) {
                const int ch0 = p * mc;
                const float* base = buf.tile_buf.data() + i + ch0 * TILE;

                float32x4_t max_vec = vld1q_f32(base);
                for (int m = 1; m < mc; ++m) {
                    float32x4_t val = vld1q_f32(base + m * TILE);
                    max_vec = vmaxq_f32(max_vec, val);
                }

                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                float32x4_t wgt_vec = vdupq_n_f32(0.0f);
                for (int m = 0; m < mc; ++m) {
                    float32x4_t val = vld1q_f32(base + m * TILE);
                    float32x4_t ev  = fast_exp_neon(vsubq_f32(val, max_vec));
                    sum_vec = vaddq_f32(sum_vec, ev);
                    wgt_vec = vmlaq_n_f32(wgt_vec, ev, static_cast<float>(m));
                }
                vst1q_f32(dfl_buf + p * 4, vdivq_f32(wgt_vec, sum_vec));
            }

            for (int k = 0; k < 4; ++k) {
                const int pxk = px + k;
                const int c = pxk % w;
                const int r = pxk / w;
                const float x1 = (c + 0.5f - dfl_buf[0 * 4 + k]) * sx;
                const float y1 = (r + 0.5f - dfl_buf[1 * 4 + k]) * sy;
                const float x2 = (c + 0.5f + dfl_buf[2 * 4 + k]) * sx;
                const float y2 = (r + 0.5f + dfl_buf[3 * 4 + k]) * sy;
                if (ts[i + k] >= conf_thresh)
                    result.push_back({x1, y1, x2, y2, ts[i + k], tc[i + k]});
            }
        }
        for (; i < tile_n; ++i) {
#else
        for (int i = 0; i < tile_n; ++i) {
#endif
            const int px = tile_start + i;
            float dfl[4];
            dfl_decode_one_scalar(buf.tile_buf.data() + i, mc, dfl);
            const int c = px % w;
            const int r = px / w;
            const float x1 = (c + 0.5f - dfl[0]) * sx;
            const float y1 = (r + 0.5f - dfl[1]) * sy;
            const float x2 = (c + 0.5f + dfl[2]) * sx;
            const float y2 = (r + 0.5f + dfl[3]) * sy;
            if (ts[i] >= conf_thresh)
                result.push_back({x1, y1, x2, y2, ts[i], tc[i]});
        }
    }
    return result;
}

// ---- DFL 解码（Tiled：消除跨 channel stride cache miss） ----
inline std::vector<float> dfl_decode(const float* data, int h, int w, int mc) {
    std::vector<float> out(4 * h * w);
    int plane = h * w, channels = 4 * mc;
    constexpr int TILE = 64;

    std::vector<float> buf(TILE * channels);

    for (int tile_start = 0; tile_start < plane; tile_start += TILE) {
        int tile_n = std::min(TILE, plane - tile_start);

        for (int ch = 0; ch < channels; ++ch) {
            const float* src = data + ch * plane + tile_start;
            float*       dst = buf.data() + ch * TILE;
            for (int i = 0; i < tile_n; ++i) dst[i] = src[i];
        }

        for (int i = 0; i < tile_n; ++i) {
            int px = tile_start + i;
            for (int p = 0; p < 4; ++p) {
                int ch0 = p * mc;
                float mv = buf[i + ch0 * TILE];
                for (int m = 1; m < mc; ++m)
                    if (buf[i + (ch0 + m) * TILE] > mv)
                        mv = buf[i + (ch0 + m) * TILE];

                float sum = 0.0f, wgt = 0.0f;
                for (int m = 0; m < mc; ++m) {
                    float ev = fast_exp(buf[i + (ch0 + m) * TILE] - mv);
                    sum += ev;
                    wgt += ev * static_cast<float>(m);
                }
                out[p * plane + px] = wgt * (1.0f / sum);
            }
        }
    }
    return out;
}

// ---- 网格坐标 → 图像坐标 ----
inline std::vector<float> box_process(const std::vector<float>& dfl_out,
                                       int h, int w, int img_h, int img_w) {
    std::vector<float> xyxy(4 * h * w);
    int plane = h * w;
    float sx = static_cast<float>(img_w) / w;
    float sy = static_cast<float>(img_h) / h;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            int px = r * w + c;
            xyxy[0*plane+px] = (c + 0.5f - dfl_out[0*plane+px]) * sx;
            xyxy[1*plane+px] = (r + 0.5f - dfl_out[1*plane+px]) * sy;
            xyxy[2*plane+px] = (c + 0.5f + dfl_out[2*plane+px]) * sx;
            xyxy[3*plane+px] = (r + 0.5f + dfl_out[3*plane+px]) * sy;
        }
    }
    return xyxy;
}

// ---- 置信度过滤 + argmax（Tiled：消除跨 class stride cache miss） ----
inline std::vector<RawBox> filter_boxes(const std::vector<float>& xyxy,
                                         const float* cls_conf,
                                         int h, int w, int num_cls,
                                         float conf_thresh) {
    int plane = h * w;
    constexpr int TILE = 64;

    std::vector<float> best_score(plane, 0.0f);
    std::vector<int>   best_cls(plane, 0);

    for (int tile = 0; tile < plane; tile += TILE) {
        int tile_n = std::min(TILE, plane - tile);

        for (int i = 0; i < tile_n; ++i) {
            int px = tile + i;
            best_score[px] = cls_conf[px];
            best_cls[px]   = 0;
        }

        for (int c = 1; c < num_cls; ++c) {
            const float* src = cls_conf + c * plane + tile;
            for (int i = 0; i < tile_n; ++i) {
                if (src[i] > best_score[tile + i]) {
                    best_score[tile + i] = src[i];
                    best_cls[tile + i]   = c;
                }
            }
        }
    }

    std::vector<RawBox> result;
    result.reserve(plane / 8);
    for (int i = 0; i < plane; ++i) {
        if (best_score[i] >= conf_thresh) {
            result.push_back({xyxy[0*plane+i], xyxy[1*plane+i],
                              xyxy[2*plane+i], xyxy[3*plane+i],
                              best_score[i], best_cls[i]});
        }
    }
    return result;
}

// ---- NMS（top-300 裁剪防 O(N²) 爆炸） ----
inline std::vector<DetectionBox> nms_boxes(const std::vector<RawBox>& raw,
                                            float nms_thresh) {
    std::vector<DetectionBox> result;
    if (raw.empty()) return result;

    // 高分截断
    constexpr int kMaxRaw = 300;
    std::vector<RawBox> top;
    if (static_cast<int>(raw.size()) > kMaxRaw) {
        top = raw;
        std::nth_element(top.begin(), top.begin() + kMaxRaw, top.end(),
                         [](const RawBox& a, const RawBox& b) {
                             return a.score > b.score;
                         });
        top.resize(kMaxRaw);
    }
    const auto& src = (static_cast<int>(raw.size()) > kMaxRaw) ? top : raw;

    std::vector<int> indices(src.size());
    for (size_t i = 0; i < src.size(); ++i) indices[i] = static_cast<int>(i);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return src[a].score > src[b].score; });

    std::vector<bool> suppressed(src.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        const auto& a = src[idx];
        result.push_back({a.x1, a.y1, a.x2, a.y2, a.score, a.cls});
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int jdx = indices[j];
            if (suppressed[jdx] || src[idx].cls != src[jdx].cls) continue;
            const auto& b = src[jdx];
            float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1) + 1e-5f);
            float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1) + 1e-5f);
            float iou_val = iw * ih /
                ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - iw * ih);
            if (iou_val > nms_thresh) suppressed[jdx] = true;
        }
    }
    return result;
}

} // namespace yolo
