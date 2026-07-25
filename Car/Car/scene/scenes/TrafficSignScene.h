#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/core/mat.hpp>

// 交通标志分岔场景
//   检测 traffic_sign → 裁剪标志图 → 调用 VLM API → 直走(左) / 右转(右)
//   三阶段状态机：APPROACH（巡线+检测）→ WAIT_VLM（停车等待）→ DRIVE（按决策走）
class TrafficSignScene : public IScene {
public:
    TrafficSignScene() = default;
    ~TrafficSignScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 三阶段状态
    enum class VlmState { APPROACH, WAIT_VLM, DRIVE };

    // 阶段1：正常巡线 + 检测交通标志 → 稳定+延时 → 发起 VLM 请求
    void handleApproach(SharedContext* ctx, const cv::Mat& seg_mask);

    // 阶段2：停车等待 VLM 决策返回
    void handleWaitVlm(SharedContext* ctx);

    // 阶段3：按 VLM 决策执行巡线
    void handleDrive(SharedContext* ctx, const cv::Mat& seg_mask);

    // 重写 onFork/onPD/onSpeed：按 fork_side 选岔路方向 + 直走/右转两套参数
    class Follower : public PathFollower {
    public:
        int fork_side = 0;  // 0=直走, 1=右转
    protected:
        void  onFork(cv::Mat& wide_mask, SharedContext* ctx) override;
        float onPD(float error, SharedContext* ctx) override;
        float onSpeed(float error, SharedContext* ctx) override;
    };

    Follower follower_;
    int traffic_sign_class_id_ = -1;

    // 状态机
    VlmState vlm_state_ = VlmState::APPROACH;
    std::shared_ptr<std::atomic<int>> vlm_result_;  // -1=进行中, 0=直走, 1=右转
    int cached_fork_side_  = 0;
    int stable_count_      = 0;
    int delay_counter_     = 0;
    std::chrono::steady_clock::time_point vlm_request_start_;
};
