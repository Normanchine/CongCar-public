#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <atomic>
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
    // VLM 请求状态
    enum class VlmState { IDLE, REQUESTING, DECIDED };

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

    // 重写 onFork：根据 VLM 结果选择保留侧
    class Follower : public PathFollower {
    public:
        int fork_side = 0;  // 0=左(直走)，1=右(右转)
    protected:
        void onFork(cv::Mat& wide_mask, SharedContext* ctx) override;
    };

    Follower follower_;
    int traffic_sign_class_id_ = -1;

    // 异步 VLM 状态（使用 detached thread + shared_ptr<atomic> 避免 future 析构阻塞）
    VlmState vlm_state_       = VlmState::IDLE;
    std::shared_ptr<std::atomic<int>> vlm_result_;  // -1=进行中, 0=直走, 1=右转
    int      cached_fork_side_ = 0;
    int      stable_count_     = 0;
};
