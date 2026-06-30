#include "CarScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <algorithm>

REGISTER_SCENE(CarScene)

void CarScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    auto det = ctx->det_result.get();

    // 找到car的class_id（只查一次）
    if (car_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "car") { car_class_id_ = i; break; }
        }
    }

    // 车检测框区域涂黑（让PathFollower绕开）
    cv::Mat mask = seg->mask.clone();
    if (det && car_class_id_ >= 0) {
        for (const auto& b : det->boxes) {
            if (b.class_id == car_class_id_ && b.confidence > 0.4f) {
                int x1 = std::max(0, static_cast<int>(b.x1));
                int y1 = std::max(0, static_cast<int>(b.y1));
                int x2 = std::min(mask.cols - 1, static_cast<int>(b.x2));
                int y2 = std::min(mask.rows - 1, static_cast<int>(b.y2));
                if (x2 > x1 && y2 > y1)
                    mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(0);
            }
        }
    }

    follower_.process(mask, ctx);
}
