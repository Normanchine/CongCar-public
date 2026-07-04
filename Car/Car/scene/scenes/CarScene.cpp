#include "CarScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <algorithm>

REGISTER_SCENE(CarScene)

void CarScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    // 找到car的class_id（只查一次）
    if (car_class_id_ < 0) {
        const auto& classes = ctx->config.ai.detection.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "car") { car_class_id_ = i; break; }
        }
    }

    // 检测车，判断在赛道哪侧 → 设置cut_side
    follower_.cut_side = -1;  // 默认无车
    auto det = ctx->det_result.get();
    if (det && car_class_id_ >= 0) {
        for (const auto& b : det->boxes) {
            if (b.class_id == car_class_id_ && b.confidence > 0.4f) {
                // 只投影车框底边两点（触地点在地面上，投影准确）
                cv::Point2f bl = ctx->perspective.toBird({b.x1, b.y2});  // 底左
                cv::Point2f br = ctx->perspective.toBird({b.x2, b.y2});  // 底右
                float car_cx = (bl.x + br.x) * 0.5f;

                // 车中心 vs 配置的中线比例 → 车在左半还是右半
                const auto& sp = ctx->config.control.scene.at("CarScene");
                float mid_x = ctx->perspective.birdW() * sp.car_mid_ratio;
                follower_.cut_side = (car_cx < mid_x) ? 0 : 1;  // 0=切左, 1=切右
                break;
            }
        }
    }

    follower_.process(seg->mask, ctx);
}

// ---- Follower::onFork：整侧切除车所在侧 + 岔路处理 ----
void CarScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    // ---- 1. 检测到车？切除车所在侧赛道 ----
    if (cut_side >= 0) {
        const auto& sp = ctx->config.control.scene.at("CarScene");
        float ratio = sp.car_cut_ratio;
        int w = wide_mask.cols;
        int cut_w = static_cast<int>(w * ratio);

        if (cut_side == 0) {
            // 车在左 → 切除左侧（车所在侧涂黑，保留右侧巡线）
            wide_mask(cv::Rect(0, 0, cut_w, wide_mask.rows)) = 0;
        } else {
            // 车在右 → 切除右侧
            wide_mask(cv::Rect(w - cut_w, 0, cut_w, wide_mask.rows)) = 0;
        }
    }

    // ---- 2. 岔路处理（有车无车都要做） ----
    PathFollower::onFork(wide_mask, ctx);
}

// ---- Follower::onPurePursuit：车体中心横向偏移 ----
bool CarScene::Follower::onPurePursuit(const cv::Mat& mask,
                                        const std::vector<cv::Point2f>& centers,
                                        float& error,
                                        cv::Point2f& aim_pt,
                                        SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp = ctrl.scene.at("CarScene");
    const auto& calib = ctx->config.calib;

    int w = mask.cols, h = mask.rows;
    float cx = w * 0.5f, cy = h * ctrl.car_cy_ratio;
    // 车体中心横向偏移（绕行时偏移车体参考点，避免擦碰）
    float offset_ratio = std::clamp(sp.car_center_x_offset_ratio, -0.45f, 0.45f);
    cx += w * offset_ratio;

    float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : sp.max_speed;
    float aim_dist = ctrl.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
    if (!pathplan::findLookAhead(centers, cx, cy, aim_dist, calib.ppm_x, calib.ppm_y, aim_pt))
        return false;

    error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                    ctrl.car_length, calib.ppm_x, calib.ppm_y)
          * ctx->steering_dir;
    return true;
}
