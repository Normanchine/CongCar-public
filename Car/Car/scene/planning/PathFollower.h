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

    // 沿边巡线方向：-1=左边界 +1=右边界 0=中点（默认左边界）
    // onCenterline 默认用 extractBoundary 取该侧边界点
    int follow_edge_ = -1;

    // 车底中心参考点横向偏移平滑（NormalScene 从画面中心平滑移到目标）
    float cur_cx_ratio_   = 0.0f;   // 当前平滑后的偏移（相对画面中心）
    int   cx_smooth_left_ = 0;      // 剩余平滑帧数

public:
    // 子类设置沿边方向
    void setFollowEdge(int edge) { follow_edge_ = edge; }

    // 查询当前沿边方向（PersonScene 起步判断用：左/右边界）
    int getFollowEdge() const { return follow_edge_; }

};
