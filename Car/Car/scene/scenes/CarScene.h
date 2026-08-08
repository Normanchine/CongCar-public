#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <opencv2/core/mat.hpp>

struct DetectionBox;  // 定义在 SharedContext.h

// 车辆避让场景（静止障碍，沿对侧边界巡线）
//   DetTask 检测到 car → 切入本场景
//   首帧锁存方向：car 框底边中点在中线左 → 沿右边界；在中线右 → 沿左边界
//   全程沿该边界巡线（不找中线），宽鸟瞰
//   退出：DetTask 连续 lost_frame_limit 帧检测不到 car 自动切回 NormalScene
//   本场景不 hold、不自己切换——退出完全交给 DetTask
class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 巡线 Follower：onCenterline 直接沿指定边界提取，不取中点
    class Follower : public PathFollower {
    public:
        int edge = 0;   // -1=沿左边界 +1=沿右边界 0=未锁存(沿中线)
    protected:
        void onCenterline(const cv::Mat& mask, std::vector<cv::Point2f>& centers,
                          SharedContext* ctx) override;
    };

    // 找置信度最高的 car 框；找到返回 true 并拷出副本
    bool findCarBox(SharedContext* ctx, DetectionBox& out);

    // 框底边中点（全图坐标）→ 减 frame_crop → toBird 鸟瞰坐标
    cv::Point2f boxToBird(const DetectionBox& box, SharedContext* ctx);

    Follower follower_;
    int      car_class_id_ = -1;
    bool     edge_locked_  = false;   // 方向只锁存一次
};
