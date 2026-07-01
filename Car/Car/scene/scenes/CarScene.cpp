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

    // 车框四角转鸟瞰，算出鸟瞰坐标系下车占区域
    follower_.car_rect = cv::Rect();
    auto det = ctx->det_result.get();
    if (det && car_class_id_ >= 0) {
        for (const auto& b : det->boxes) {
            if (b.class_id == car_class_id_ && b.confidence > 0.4f) {
                // 车底中心 + 四个角 → 鸟瞰 → 取包围盒
                std::vector<cv::Point2f> corners = {
                    {b.x1, b.y1}, {b.x2, b.y1}, {b.x2, b.y2}, {b.x1, b.y2}
                };
                std::vector<cv::Point2f> bird_pts;
                for (auto& c : corners)
                    bird_pts.push_back(ctx->perspective.toBird(c));

                float min_x = bird_pts[0].x, max_x = bird_pts[0].x;
                float min_y = bird_pts[0].y, max_y = bird_pts[0].y;
                for (auto& pt : bird_pts) {
                    if (pt.x < min_x) min_x = pt.x;
                    if (pt.x > max_x) max_x = pt.x;
                    if (pt.y < min_y) min_y = pt.y;
                    if (pt.y > max_y) max_y = pt.y;
                }
                follower_.car_rect = cv::Rect(
                    static_cast<int>(min_x), static_cast<int>(min_y),
                    static_cast<int>(max_x - min_x + 1),
                    static_cast<int>(max_y - min_y + 1));
                // 宽鸟瞰→裁切中心偏移（process()内会crop）
                int crop_off = (ctx->perspective.birdW()
                                - ctx->config.calib.bird_width) / 2;
                follower_.car_rect.x -= crop_off;
                break;  // 只处理第一个car
            }
        }
    }

    follower_.process(seg->mask, ctx);
}

// ---- CarFollower::onCenterline：利用车框坐标跳过一个边界扫描 ----
void CarScene::Follower::onCenterline(const cv::Mat& mask,
                                       std::vector<cv::Point2f>& centers,
                                       SharedContext* ctx) {
    if (car_rect.width <= 0 || car_rect.height <= 0) {
        PathFollower::onCenterline(mask, centers, ctx);
        return;
    }

    int w = mask.cols, h = mask.rows;
    centers.clear();
    centers.reserve(h);

    // 用车框前的行判断车在赛道哪侧：车中X vs 一行轨道中X
    float car_cx = car_rect.x + car_rect.width * 0.5f;
    int ref_row = std::max(car_rect.y - 1, 0);
    const uint8_t* ref = mask.ptr<uint8_t>(ref_row);
    int ref_l = -1, ref_r = -1;
    for (int c = 0; c < w && ref_l < 0; ++c) if (ref[c] > 0) ref_l = c;
    for (int c = w - 1; c >= 0 && ref_r < 0; --c) if (ref[c] > 0) ref_r = c;
    float track_cx = (ref_l >= 0 && ref_r >= 0) ? (ref_l + ref_r) * 0.5f : w * 0.5f;
    bool car_on_left = (car_cx < track_cx);  // 车在左 → 从右边绕

    for (int r = 0; r < h; ++r) {
        const uint8_t* row = mask.ptr<uint8_t>(r);

        int left = -1, right = -1;
        if (r >= car_rect.y && r < car_rect.y + car_rect.height) {
            if (car_on_left) {
                // 车在左 → 左边界从车右侧向外扫，右边界正常
                for (int c = car_rect.x + car_rect.width; c < w; ++c) {
                    if (row[c] > 0) { left = c; break; }
                }
                if (left < 0) continue;
                for (int c = w - 1; c > left; --c) {
                    if (row[c] > 0) { right = c; break; }
                }
            } else {
                // 车在右 → 右边界从车左侧向内扫，左边界正常
                for (int c = car_rect.x - 1; c >= 0; --c) {
                    if (row[c] > 0) { right = c; break; }
                }
                if (right < 0) continue;
                for (int c = 0; c < right; ++c) {
                    if (row[c] > 0) { left = c; break; }
                }
            }
        } else {
            // 非车行：正常扫描
            for (int c = 0; c < w; ++c) {
                if (row[c] > 0) { left = c; break; }
            }
            if (left < 0) continue;
            for (int c = w - 1; c >= left; --c) {
                if (row[c] > 0) { right = c; break; }
            }
        }
        if (left < 0 || right < 0) continue;

        centers.emplace_back((left + right) * 0.5f, static_cast<float>(r));
    }

    pathplan::smoothCenterline(centers);
}
