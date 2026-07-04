#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 重写onFork：在宽鸟瞰mask上整侧切除车所在侧，然后正常巡线
    // 重写onPurePursuit：车体中心横向偏移
    class Follower : public PathFollower {
    public:
        int cut_side = -1;  // -1=无车, 0=车在左切左, 1=车在右切右
    protected:
        void onFork(cv::Mat& wide_mask, SharedContext* ctx) override;
        bool onPurePursuit(const cv::Mat& mask,
                           const std::vector<cv::Point2f>& centers,
                           float& error,
                           cv::Point2f& aim_pt,
                           SharedContext* ctx) override;
    };

    Follower follower_;
    int car_class_id_ = -1;
};
