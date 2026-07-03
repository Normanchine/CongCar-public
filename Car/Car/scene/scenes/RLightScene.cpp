#include "RLightScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <algorithm>
#include <cmath>

//REGISTER_SCENE(RLightScene)

void RLightScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // 找到 R_Light 的 class_id（只查一次）
    if (r_light_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "R_Light") {
                r_light_class_id_ = i;
                break;
            }
        }
    }

    // ---- 1. 正常巡线 ----
    follower_.process(seg->mask, ctx);

    // ---- 2. 红灯停车逻辑 ----
    if (r_light_class_id_ < 0) return;  // 类别不存在，跳过

    auto det = ctx->det_result.get();
    if (!det) return;

    // 在当前帧中找红灯检测框
    float light_y = -1.0f;
    for (const auto& b : det->boxes) {
        if (b.class_id == r_light_class_id_ && b.confidence > 0.3f) {
            // 取检测框底部中心点（红灯最下端）
            float cx = (b.x1 + b.x2) * 0.5f;
            float bottom_y = b.y2;

            // 转换到鸟瞰图坐标
            cv::Point2f bird_pt = ctx->perspective.toBird(cv::Point2f(cx, bottom_y));

            // 裁切中心区域的偏移
            int crop_off = (ctx->perspective.birdW()
                            - ctx->config.calib.bird_width) / 2;
            bird_pt.x -= crop_off;

            // 只取在鸟瞰图范围内且在车前方的红灯
            if (bird_pt.y > 0 && bird_pt.y < ctx->config.calib.bird_height) {
                light_y = std::max(light_y, bird_pt.y);
            }
        }
    }

    if (light_y < 0) return;  // 没有有效红灯

    // ---- 3. 计算停车阈值 ----
    const auto& sp = ctx->config.control.scene.at("RLightScene");
    float ppm_y = ctx->config.calib.ppm_y;

    // stop_y = 基准值(像素) + 速度系数 * max_speed * ppm_y
    float stop_y = sp.r_light_stop_y_cm / 100.0f * ppm_y
                 + sp.r_light_stop_speed_k * sp.max_speed * ppm_y;

    // ---- 4. 触发停车 ----
    if (light_y >= stop_y) {
        ctx->target_speed.store(0.0f);
    }
}
