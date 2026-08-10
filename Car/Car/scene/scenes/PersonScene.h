#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <opencv2/core/mat.hpp>

struct DetectionBox;  // 定义在 SharedContext.h

// 行人避让场景（动态行人，停车等待 + 让路起步）
//   DetTask 检测到 person → 切入本场景
//   停车：行人底部（鸟瞰坐标 y）到达鸟瞰图高度 person_stop_ratio（默认0.5）→
//         场景内直发停车指令（不溜车）
//   起步（人让开路才走）：
//     左边界：人向右走 且 人已过赛道中心（x > 中心 x）
//     右边界：人向左走 且 人已过赛道中心（x < 中心 x）
//   退出：DetTask 连续 lost_frame_limit 帧检测不到 person 自动切回 NormalScene
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 找置信度最高的 person 框；找到返回 true 并拷出副本
    bool findPersonBox(SharedContext* ctx, DetectionBox& out);

    // 框底边中点（全图坐标）→ 减 frame_crop → toBird 鸟瞰坐标
    cv::Point2f boxToBird(const DetectionBox& box, SharedContext* ctx);

    // 宽鸟瞰赛道中线在 bird_y 处的 x（插值）；中线为空用宽鸟瞰中心兜底
    float trackCenterXAt(const cv::Mat& seg_mask, float bird_y, SharedContext* ctx);

    // 更新行人左右移动方向（逐帧 x 变化）
    void updateMotion(const cv::Point2f& obs);

    // 起步判断：按当前边界 + 行人方向 + 已过赛道中心
    bool shouldGo(const cv::Point2f& obs, float track_center_x) const;

    PathFollower follower_;
    int person_class_id_ = -1;

    bool  stopped_      = false;     // 停车等待中
    bool  moving_right_ = false;     // 行人向右移动
    bool  moving_left_  = false;     // 行人向左移动
    cv::Point2f prev_obs_{0, 0};     // 上一帧行人鸟瞰位置（方向判断）
    bool  has_prev_     = false;
    int   lost_person_frames_ = 0;   // 行人丢失帧计数（防检测抖动前冲）
};
