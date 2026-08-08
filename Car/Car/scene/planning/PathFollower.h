#pragma once

#include "PathPlanning.h"
#include "../../common/PerspectiveMapper.h"
#include "../../config/Config.h"
#include <opencv2/imgproc.hpp>
#include <vector>

struct SharedContext;

class PathFollower {
public:
    virtual ~PathFollower() = default;

    // 模板方法：每帧调用，流程固定不可重写
    // 返回 false 表示停车保护触发
    bool process(const cv::Mat& seg_mask, SharedContext* ctx);

protected:
    // ---- 可重写步骤（子类只改需要改的） ----

    // 岔路处理：默认调 handleFork(wide_mask, fork_state_, fork.*)
    virtual void onFork(cv::Mat& wide_mask, SharedContext* ctx);

    // 中线提取：默认 extract + smooth
    virtual void onCenterline(const cv::Mat& mask,
                              std::vector<cv::Point2f>& centers,
                              SharedContext* ctx);

    // 纯跟踪 + 航向误差：默认计算车体中心、动态前视、寻找前视点、计算航向误差
    // 返回 false 表示找不到前视点
    virtual bool onPurePursuit(const cv::Mat& mask,
                               const std::vector<cv::Point2f>& centers,
                               float& error,
                               cv::Point2f& aim_pt,
                               SharedContext* ctx);

    // 丢线容错：默认盲开+速度衰减（保持方向 + 逐帧递增偏移 → 超限停车）
    virtual void onLost(SharedContext* ctx);

    // PD 控制：默认返回 pwm 值
    virtual float onPD(float error, SharedContext* ctx);

    // 速度策略：默认大弯减速+加速缓冲
    virtual float onSpeed(float error, SharedContext* ctx);

    // ---- 内部状态，子类可访问 ----
    float last_error_  = 0.0f;
    float last_pwm_    = 1500.0f;
    float last_speed_  = 0.0f;
    int   lost_frames_ = 0;
    bool  first_frame_ = true;
    pathplan::ForkState fork_state_;  // 岔路持久化状态（场景切换自动重置）

    // 巡线裁切宽度倍数（默认1=裁bird_width中心区；CarScene设bird_width_scale=全宽鸟瞰）
    // 宽鸟瞰巡线：沿边界走时远处赛道边界不脱离视野，避免丢线
    float crop_scale_ = 1.0f;

};
