#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <opencv2/core/mat.hpp>

struct DetectionBox;  // 定义在 SharedContext.h

// 车辆避让场景（静止障碍，沿对侧边界巡线）
//   DetTask 检测到 car → 切入本场景
//   实时检测方向：每帧 car 框底边中点 vs 赛道中心，在左 → 沿右边界；在右 → 沿左边界
//   全程沿该边界巡线（不找中线），宽鸟瞰
//   退出：DetTask 连续 lost_frame_limit 帧检测不到 car 自动切回 NormalScene
//   本场景不 hold、不自己切换——退出完全交给 DetTask
class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 巡线 Follower：直接用基类 setFollowEdge 切边界，无需重写 onCenterline
    class Follower : public PathFollower {};

    // 找置信度最高的 car 框；找到返回 true 并拷出副本
    bool findCarBox(SharedContext* ctx, DetectionBox& out);

    // 框底边中点（全图坐标）→ 减 frame_crop → toBird 鸟瞰坐标
    cv::Point2f boxToBird(const DetectionBox& box, SharedContext* ctx);

    // 宽鸟瞰赛道中线在 bird_y 处的 x（插值）；中线为空用宽鸟瞰中心兜底
    float trackCenterXAt(const cv::Mat& seg_mask, float bird_y, SharedContext* ctx);

    Follower follower_;
    int      car_class_id_ = -1;
};
