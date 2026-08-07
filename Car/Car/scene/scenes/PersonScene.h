#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <vector>

// 行人绕行场景（引导线固定 + 车移动坐标系）
//   APPROACH：正常巡线 + 监控人到鸟瞰图纵向中点
//   HOLDING：触发 → 中线等比平移生成引导线（固定坐标系）→ hold() →
//            车由 yaw 驱动在坐标系上移动 → 两段纯跟踪（锁M→锁E）
//   DONE：车越过终点 → release() → 切回 NormalScene
//
//   引导线 = 触发帧中线逐点等比平移（生成后固定不动）：
//     S（起点，y=car_cy，偏移=0）→ M（拐点，y=person.y，对侧偏移0.4m）→ E（终点，y=中线最远点，偏移=0）
//   段1 锁M快速建立横向分离，段2 锁E平滑汇入中线
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    enum class State { APPROACH, HOLDING, DONE };

    // 阶段1：正常巡线 + 检测人 → 人到鸟瞰图纵向中点时触发 → 生成引导线
    void handleApproach(SharedContext* ctx, const cv::Mat& seg_mask);

    // 阶段2：车由 yaw 驱动在引导线坐标系上移动 → 纯跟踪锁 M/E
    void handleHolding(SharedContext* ctx);

    // 阶段3：释放 hold，切回正常场景
    void handleDone(SharedContext* ctx);

    // 在宽鸟瞰图 mask 上提取中线 → 逐点等比平移 → 生成引导线
    // 顺带记录 M_（拐点）和 E_（终点），返回 false 表示中线为空
    bool generateGuideLine(const cv::Mat& wide_mask,
                           const cv::Point2f& person_bird,
                           SharedContext* ctx);

    PathFollower follower_;           // APPROACH 阶段正常巡线
    State state_ = State::APPROACH;

    // ---- 引导线（触发帧生成一次，固定坐标系） ----
    std::vector<cv::Point2f> guide_pts_;  // 引导线点集（固定，不滚动）
    cv::Mat  guide_mask_;                 // 触发帧宽鸟瞰赛道mask（供DisplayTask画底图）
    cv::Point2f M_ = {0, 0};              // 拐点（固定）
    cv::Point2f E_ = {0, 0};              // 终点（固定）

    // ---- 车在引导线坐标系上的位置（HOLDING 逐帧由 yaw 驱动更新） ----
    float car_x_ = 0.0f;
    float car_y_ = 0.0f;
    float car_x_init_ = 0.0f;   // 入口位置（坐标系原点）
    float car_y_init_ = 0.0f;
    std::chrono::steady_clock::time_point last_tp_;

    // ---- IMU 航向角校准 ----
    float entry_yaw_ = 0.0f;
    float rel_yaw_   = 0.0f;

    // ---- 纯跟踪状态 ----
    float last_error_     = 0.0f;
    float last_speed_     = 0.0f;
    bool  segment1_first_ = true;  // 段1 首帧：继承ctx状态 + PD跳过D项
    bool  segment2_first_ = true;  // 段2 首帧：防切段时error跳变

    // ---- 缓存 ----
    int person_class_id_ = -1;
};
