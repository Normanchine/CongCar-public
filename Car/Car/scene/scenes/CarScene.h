#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

// 车辆避让场景
//   检测到 car → 判断车在赛道左侧/右侧 → 切除车所在区域 → 另一侧膨胀补偿
//   全程使用宽鸟瞰巡线（跳过crop），补偿后赛道宽度不变，中线自然偏移引导绕行
class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 重写onFork/onCenterline/onPurePursuit：宽鸟瞰全流程
    class Follower : public PathFollower {
    public:
        int cut_side = -1;     // -1=无车, 0=车在左切左, 1=车在右切右
        int cut_width_px = 0;  // 切除宽度（像素，=检测框鸟瞰宽度 + margin）
    protected:
        void  onFork(cv::Mat& wide_mask, SharedContext* ctx) override;
        void  onCenterline(const cv::Mat& mask,
                           std::vector<cv::Point2f>& centers,
                           SharedContext* ctx) override;
        bool  onPurePursuit(const cv::Mat& mask,
                            const std::vector<cv::Point2f>& centers,
                            float& error,
                            cv::Point2f& aim_pt,
                            SharedContext* ctx) override;
    private:
        cv::Mat wide_mask_;  // 宽鸟瞰副本，onCenterline/onPurePursuit 使用
    };

    Follower follower_;
    int car_class_id_ = -1;
    bool cutting_active_ = false;  // 一旦触发就保持，场景退出时重置
};
