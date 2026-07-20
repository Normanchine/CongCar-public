#include "CarScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <algorithm>

REGISTER_SCENE(CarScene)

void CarScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // 找car class_id（只查一次）
    if (car_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "car") { car_class_id_ = i; break; }
        }
    }

    // 已触发？保持切除，不再看距离和检测结果
    if (!cutting_active_) {
        follower_.cut_side = -1;
        follower_.cut_width_px = 0;

        auto det = ctx->det_result.get();
        if (det && car_class_id_ >= 0 && ctx->perspective.isReady()) {
            const auto& ctrl = ctx->config.control;
            const auto& sp = ctrl.scene.at("CarScene");
            const auto& calib = ctx->config.calib;
            float crop_y = ctx->config.frame_crop.enable
                ? static_cast<float>(ctx->config.frame_crop.y) : 0.0f;
            int margin = sp.car_side_margin;

            // 找最近的car（y2最大 = 离车最近）
            const DetectionBox* best = nullptr;
            for (const auto& b : det->boxes) {
                if (b.class_id == car_class_id_ && b.confidence > 0.4f) {
                    if (!best || b.y2 > best->y2) best = &b;
                }
            }

            if (best) {
                // 框底边两点 → 鸟瞰坐标
                cv::Point2f bl = ctx->perspective.toBird({best->x1, best->y2 - crop_y});
                cv::Point2f br = ctx->perspective.toBird({best->x2, best->y2 - crop_y});
                float car_cx = (bl.x + br.x) * 0.5f;
                float car_width = br.x - bl.x;
                float car_bird_y = (bl.y + br.y) * 0.5f;

                // 纵向距离判断：够近才触发，一旦触发锁住
                float my_y = calib.bird_height * ctrl.car_cy_ratio;
                float dist_m = (my_y - car_bird_y) / calib.ppm_y;
                float cur_spd = ctx->target_speed.load(std::memory_order_acquire);
                float trigger = sp.car_trigger_dist_m + cur_spd * sp.car_trigger_speed_k;
                if (dist_m > 0 && dist_m < trigger) {
                    float mid_x = ctx->perspective.birdW() * 0.5f;
                    follower_.cut_side = (car_cx < mid_x) ? 0 : 1;
                    follower_.cut_width_px = static_cast<int>(car_width) + margin;
                    cutting_active_ = true;  // 锁住，后续帧不再判断
                }
            }
        }
    }

    follower_.process(seg->mask, ctx);
}

// ---- 切车所在侧 + 另一侧膨胀补偿 ----
void CarScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    if (cut_side >= 0 && cut_width_px > 0) {
        int w = wide_mask.cols;

        // ---- 1. 切除车所在侧 ----
        if (cut_side == 0) {
            wide_mask(cv::Rect(0, 0, cut_width_px, wide_mask.rows)) = 0;  // 车在左，切左
        } else {
            wide_mask(cv::Rect(w - cut_width_px, 0, cut_width_px, wide_mask.rows)) = 0;  // 车在右，切右
        }

        // ---- 2. 另一侧膨胀补偿（水平方向，切多少补多少） ----
        int ks = cut_width_px * 2 + 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ks, 1));
        cv::dilate(wide_mask, wide_mask, kernel);
    }

    // ---- 3. 岔路处理 ----
    PathFollower::onFork(wide_mask, ctx);

    // ---- 4. 保存宽鸟瞰副本，供 onCenterline/onPurePursuit 使用 ----
    wide_mask_ = wide_mask;
}

// ---- 中线提取：跳过crop，直接用宽鸟瞰 ----
void CarScene::Follower::onCenterline(const cv::Mat& /*cropped*/,
                                       std::vector<cv::Point2f>& centers,
                                       SharedContext* /*ctx*/) {
    pathplan::extractCenterline(wide_mask_, centers);
    pathplan::smoothCenterline(centers);
}

// ---- 纯跟踪：车底中心用宽鸟瞰坐标系 ----
bool CarScene::Follower::onPurePursuit(const cv::Mat& /*cropped*/,
                                        const std::vector<cv::Point2f>& centers,
                                        float& error,
                                        cv::Point2f& aim_pt,
                                        SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at("CarScene");
    const auto& calib = ctx->config.calib;

    int w = wide_mask_.cols, h = wide_mask_.rows;
    float cx = w * 0.5f, cy = h * ctrl.car_cy_ratio;
    float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : sp.max_speed;
    float aim_dist = sp.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_dist, calib.ppm_x, calib.ppm_y, aim_pt))
        return false;

    error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                    ctrl.car_length, calib.ppm_x, calib.ppm_y)
          * ctx->steering_dir;
    return true;
}
