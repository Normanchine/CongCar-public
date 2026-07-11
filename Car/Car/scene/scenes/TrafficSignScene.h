#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/core/mat.hpp>

// 交通标志分岔场景
//   检测 traffic_sign → 裁剪标志图 → 调用千帆视觉大模型 → 直走(左) / 右转(右)
//   使用异步状态机避免阻塞控制循环：IDLE → REQUESTING → DECIDED
class TrafficSignScene : public IScene {
public:
    TrafficSignScene() = default;
    ~TrafficSignScene() override = default;

    void run(SharedContext* ctx) override;

private:
    // VLM 三阶段状态机：APPROACH（巡线+检测）→ WAIT_VLM（停车等待）→ DRIVE（按决策走）
    enum class VlmState { APPROACH, WAIT_VLM, DRIVE };

    // 准备发送给 VLM 的图片：裁剪 + 扩边 + 缩放 → JPEG 字节
    static std::vector<uint8_t> prepareSignImage(const cv::Mat& frame,
                                                  float x1, float y1,
                                                  float x2, float y2,
                                                  const VlmConfig& cfg);

    // 同步调用千帆 VLM API（在后台线程中执行），返回 0=直走 / 1=右转
    static int callVlmApi(const std::vector<uint8_t>& jpeg_bytes,
                          const VlmConfig& cfg);

    // 清理 VLM 输出文本
    static std::string cleanOutput(const std::string& text);

    // 重写 onPD/onSpeed：按 fork_side 选直走/右转两套参数，其余与 PathFollower 一致
    class Follower : public PathFollower {
    public:
        int fork_side = 0;  // 0=直走, 1=右转
    protected:
        float onPD(float error, SharedContext* ctx) override;
        float onSpeed(float error, SharedContext* ctx) override;
    };

    Follower follower_;   // 阶段1/阶段3 共用（纯 PathFollower，无锁舵降速）
    int traffic_sign_class_id_ = -1;

    // 异步 VLM 状态（使用 detached thread + shared_ptr<atomic> 避免 future 析构阻塞）
    VlmState vlm_state_       = VlmState::APPROACH;
    std::shared_ptr<std::atomic<int>> vlm_result_;  // -1=进行中, 0=直走, 1=右转
    int      cached_fork_side_ = 0;
    int      stable_count_     = 0;
    int      delay_counter_    = 0;  // 稳定检测后的帧延迟计数
    std::chrono::steady_clock::time_point vlm_request_start_;
};
