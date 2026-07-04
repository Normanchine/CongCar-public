#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

// 行人避让场景 — 简化状态机
//   DRIVING（默认）→ STOPPING：人进入纵向危险距离（每次进入场景仅触发一次）
//   STOPPING → DRIVING：人消失 或 人过赛道中线 N 次
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 重写onSpeed：停车后的加速恢复逻辑
    class Follower : public PathFollower {
    protected:
        float onSpeed(float error, SharedContext* ctx) override;
    };

    Follower follower_;

    int person_class_id_ = -1;
    bool has_stopped_ = false;     // 本次进入场景是否已停过车（只停一次）
    int  cross_count_ = 0;         // 人过赛道中线次数
    bool was_on_track_ = false;    // 上一帧人是否在赛道上（上升沿检测）

    void stopCar(SharedContext* ctx);
    void followLine(SharedContext* ctx);

    // 查找最近行人脚底点（原图坐标），返回是否检测到
    bool findPersonFoot(cv::Point2f& foot_img, SharedContext* ctx) const;

    // 判断人是否在纵向危险距离内（鸟瞰）
    bool isPersonInDanger(cv::Point2f foot_img, SharedContext* ctx) const;

    // 判断人脚底是否踩在赛道mask上（原图坐标）
    bool isFootOnTrack(cv::Point2f foot_img, const cv::Mat& seg_mask) const;
};
