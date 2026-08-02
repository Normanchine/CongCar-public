#pragma once

#include "../framework/IScene.h"
#include "../planning/PathFollower.h"
#include "../../config/Config.h"
#include <opencv2/core/mat.hpp>
#include <chrono>
#include <vector>

// 行人绕行场景
//   APPROACH：正常巡线 + 监控人框纵向距离
//   HOLDING：到达临界距离 → 生成弧形绕行地图 → hold() → 沿弧线走 + 航位推算
//   DONE：弧线走完 → release() → 切回 NormalScene
class PersonScene : public IScene {
public:
    PersonScene() = default;
    ~PersonScene() override = default;

    void run(SharedContext* ctx) override;

private:
    enum class State { APPROACH, HOLDING, DONE };

    // 阶段1：正常巡线 + 检测人 → 纵向距离判定 → 触发生成弧线
    void handleApproach(SharedContext* ctx, const cv::Mat& seg_mask);

    // 阶段2：每帧移位弧线 → 逆变换回图像坐标 → 巡线 + 航位推算
    void handleHolding(SharedContext* ctx);

    // 阶段3：释放 hold，切回正常场景
    void handleDone(SharedContext* ctx);

    // 在鸟瞰图坐标系生成贝塞尔弧线（存 arc_bezier_pts_ / arc_thickness_px_ / arc_total_length_m_）
    // 返回 false 表示生成失败（中线为空等）
    bool generateArc(const cv::Mat& wide_mask,
                     const cv::Point2f& person_bird,
                     SharedContext* ctx);

    PathFollower follower_;
    State state_ = State::APPROACH;

    // ---- 弧线数据（鸟瞰图坐标，触发时生成一次） ----
    std::vector<cv::Point2f> arc_bezier_pts_;  // 贝塞尔曲线采样点（100个）
    int   arc_thickness_px_ = 0;               // 弧线绘制宽度（= 原赛道宽度，像素）
    float arc_total_length_m_ = 0.0f;          // 弧线总长（米），用于判定到达

    // ---- 航位推算（HOLDING 阶段逐帧累加） ----
    float traveled_m_ = 0.0f;
    std::chrono::steady_clock::time_point last_tp_;

    // ---- 缓存 person 类别在 classes 向量中的索引 ----
    int person_class_id_ = -1;
};
