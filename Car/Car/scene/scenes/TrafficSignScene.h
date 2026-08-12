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
//   四阶段状态机：APPROACH（巡线+检测）→ BACKUP（停车后倒车退回）→ WAIT_VLM（停车等待）→ DRIVE（按决策走）
class TrafficSignScene : public IScene {
public:
    TrafficSignScene() = default;
    ~TrafficSignScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // 四阶段状态
    enum class VlmState { APPROACH, BACKUP, WAIT_VLM, DRIVE };

    // 阶段1：正常巡线 + 检测交通标志 → 稳定+延时 → 发起 VLM 请求
    void handleApproach(SharedContext* ctx, const cv::Mat& seg_mask);

    // 阶段1.5：停车后倒车退回（惯性冲过头，倒车 backup_frames 帧退到合适位置）
    void handleBackup(SharedContext* ctx);

    // 停车 + 进入倒车阶段（倒车 backup_frames 帧，倒完按 vlm_requested_ 分流到 WAIT_VLM/DRIVE）
    void beginBackup(SharedContext* ctx);

    // 阶段2：停车等待 VLM 决策返回
    void handleWaitVlm(SharedContext* ctx);

    // 阶段3：按 VLM 决策执行巡线
    void handleDrive(SharedContext* ctx, const cv::Mat& seg_mask);

    // 重写 onCenterline/onFork：按 fork_side 选边界方向（PD/速度用基类全局参数）
    class Follower : public PathFollower {
    public:
        int fork_side = 1;  // 沿边方向：1=右边界（进入场景默认右，路牌在右侧）；0=左边界（VLM决策直走）

        // 右转盲开状态（DRIVE 右转：固定速度 + 全程右满舵，走够帧数后恢复巡线）
        int  turn_frames_left_ = 0;      // 剩余盲开帧数
        bool turn_blind_done_  = false;  // 右转盲开是否已完成（完成后不再重复盲开）

        // 右转盲开一帧；返回 true=仍在盲开中，false=已走完恢复巡线
        bool blindTurnRight(int frames, float speed, SharedContext* ctx);

    protected:
        void  onCenterline(const cv::Mat& mask, std::vector<cv::Point2f>& centers,
                           SharedContext* ctx) override;
        void  onFork(cv::Mat& wide_mask, SharedContext* ctx) override;
    };

    Follower follower_;
    int traffic_sign_class_id_ = -1;

    // 状态机
    VlmState vlm_state_ = VlmState::APPROACH;
    std::shared_ptr<std::atomic<int>> vlm_result_;  // -1=进行中, 0=直走, 1=右转
    int cached_fork_side_  = 0;
    int stable_count_      = 0;
    int delay_counter_     = 0;
    int backup_frames_left_ = 0;   // 剩余倒车帧数（BACKUP 阶段每帧递减）
    bool vlm_requested_    = false;  // 本次进场景是否发起了 VLM 请求（BACKUP 结束后分流用）
    std::chrono::steady_clock::time_point vlm_request_start_;
};
