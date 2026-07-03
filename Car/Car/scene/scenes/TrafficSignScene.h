#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include <opencv2/core/mat.hpp>

class TrafficSignScene : public IScene {
public:
    TrafficSignScene() = default;
    ~TrafficSignScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // OCR 函数：0=直走，1=右转
    static int recognizeTrafficSign(const cv::Mat& sign_image);

    // 重写 onFork 的 Follower
    class Follower : public PathFollower {
    public:
        int fork_side = 0;  // 0=左(直走)，1=右(右转)
    protected:
        void onFork(cv::Mat& wide_mask, SharedContext* ctx) override;
    };

    Follower follower_;
    int traffic_sign_class_id_ = -1;  // 交通标志类别ID（缓存）
};
