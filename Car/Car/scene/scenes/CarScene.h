#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"

class CarScene : public IScene {
public:
    CarScene() = default;
    ~CarScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 重写onCenterline：跳过车区像素，无需修改mask
    class Follower : public PathFollower {
    public:
        cv::Rect car_rect;  // 鸟瞰坐标（空rect不跳过）
    protected:
        void onCenterline(const cv::Mat& mask, std::vector<cv::Point2f>& centers,
                          SharedContext* ctx) override;
    };

    Follower follower_;
    int car_class_id_ = -1;
};
