#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>
#include "../common/FrameQueue.h"
#include "../common/PerfStats.h"
#include "../common/SharedResult.h"
#include "../common/PerspectiveMapper.h"
#include "../config/Config.h"
#include "../scene/framework/SceneManager.h"
#include <opencv2/core/mat.hpp>
#include "../threadpool/ThreadPool.h"

class UartIO;  // 前向声明，指针由ControlTask注入

// ---- 语义分割共享结果 ----
struct SegResult {
    cv::Mat   mask;        // (H,W) CV_8UC1，像素值=类别索引
    int       width  = 0;
    int       height = 0;
    uint64_t  frame_id = 0;
};

// ---- 目标检测共享结果 ----
struct DetectionBox {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float confidence = 0;
    int   class_id   = 0;
};

struct DetResult {
    std::vector<DetectionBox> boxes;
    int       width  = 0;
    int       height = 0;
    uint64_t  frame_id = 0;
};

// ---- 线程间共享的所有变量集中存放 ----
struct SharedContext {
    // ---- 三个共享队列 ----
    FrameQueue<cv::Mat> display_queue{3};
    FrameQueue<cv::Mat> detection_queue{3};
    FrameQueue<cv::Mat> segmentation_queue{3};

    // ---- 场景管理器 ----
    SceneManager scene_manager;

    // ---- 线程池（调试模式启用，比赛模式为nullptr） ----
    ThreadPool* thread_pool = nullptr;

    // ---- 全局配置 ----
    AppConfig config;

    // ---- 性能统计 ----
    PerfStats perf_stats;

    // ---- AI推理共享结果（多读单写，双缓冲零锁） ----
    SharedResult<SegResult> seg_result;
    SharedResult<DetResult> det_result;

    // ---- 鸟瞰图中线调试数据（NormalScene写，DisplayTask读） ----
    std::mutex               bird_mtx_;
    cv::Mat                  bird_mask_;
    std::vector<cv::Point2f> bird_centers_;
    cv::Point2f              bird_aim_pt_{0, 0};
    bool                     bird_valid_ = false;

    // ---- 全局坐标系元数据（PathFollower 每帧写） ----
    struct GuideMeta {
        cv::Point2f origin  = {0, 0};  // 坐标系原点（宽鸟瞰坐标）
        cv::Point2f e_pt    = {0, 0};  // 当前帧中线最远点（宽鸟瞰坐标，盲开时冻结）
        cv::Point2f car_pt  = {0, 0};  // 车在坐标系中的实时位置
        float       car_yaw = 0.0f;    // 车朝向（度，rel_yaw）
        bool        blind   = false;   // 盲开标记
    };
    std::mutex  guide_mtx_;
    GuideMeta   guide_meta_;

    // ---- 控制输出 ----
    std::atomic<float> steering_pwm{1500};   // 舵机PWM (500-2500, 中值1500)
    std::atomic<float> target_speed{0};      // 目标速度
    std::atomic<float> last_error{0};        // 跨场景error继承（防PD脉冲）
    float steering_dir = 1.0f;               // 转向方向（AR图像左右镜像时置-1）

    // ---- CarScene 沿边方向（CarScene写，DisplayTask读，HUD显示） ----
    // -1=沿左边界 0=未锁存(沿中线) +1=沿右边界
    std::atomic<int> car_edge{0};

    // ---- 当前巡线参考点横向偏移比例（PathFollower写，DisplayTask读，画蓝点） ----
    // 控制器每帧的目标偏移（-follow_edge_*lateral_offset_ratio），跨场景一致。
    std::atomic<float> ref_ratio{0.0f};

    // ---- CarScene 障碍鸟瞰点（CarScene写，DisplayTask读，鸟瞰图画点） ----
    // car_obs_pt：障碍框底边中点的宽鸟瞰坐标；car_obs_valid：本帧是否有检测
    std::mutex    car_obs_mtx_;
    cv::Point2f   car_obs_pt_{0, 0};
    bool          car_obs_valid_ = false;

    // ---- PersonScene 行人鸟瞰点（PersonScene写，DisplayTask读，鸟瞰图画十字） ----
    // person_obs_pt：行人框底边中点的宽鸟瞰坐标；person_obs_valid：本帧是否有检测
    std::mutex    person_obs_mtx_;
    cv::Point2f   person_obs_pt_{0, 0};
    bool          person_obs_valid_ = false;

    // ---- UART直发（场景紧急停车可绕过ControlTask下发循环） ----
    UartIO* uart_io = nullptr;

    // ---- 透视变换 ----
    PerspectiveMapper perspective;

    // ---- 工作帧（CaptureTask 写入，场景裁剪用） ----
    FrameQueue<cv::Mat> work_frame_queue{3};

    // ---- 工作帧尺寸（CaptureTask 写入，DetectionTask 读取用于坐标映射） ----
    std::atomic<int> frame_width{0};
    std::atomic<int> frame_height{0};

    // ---- 全图尺寸（CaptureTask 写入，检测走全图时供 DetTask 坐标映射） ----
    std::atomic<int> full_frame_width{0};
    std::atomic<int> full_frame_height{0};


    // ---- VLM 状态（TrafficSignScene 写，DisplayTask 读） ----
    std::atomic<int>     vlm_state{0};        // 0=IDLE, 1=REQUESTING, 2=DECIDED
    std::atomic<int>     vlm_decision{-1};     // -1=未决策, 0=直走, 1=右转
    std::atomic<int64_t> vlm_latency_ms{0};   // 最近一次 VLM 请求耗时(ms)

    // ---- 全局交通标志决策（TrafficSignScene 写，跨场景持久，全赛只判断一次） ----
    // 赛事全流程只有一圈走右转岔路，所以只需 VLM 判断一次。
    // 存的是"下一次进场景要走的方向" = 本次 VLM 结果的反值（本次右转→全局直走）：
    // -1=尚未决策（下次进 TrafficSignScene 会发起 VLM），0=直走，1=右转（后续进场景直接读它当 fork_side）
    std::atomic<int> traffic_decision{-1};

    // ---- 全局右转锁（TrafficSignScene 写，ControlTask 每帧递减，>0 表示锁定中） ----
    // VLM 返回"右转"时置 right_turn_lock_seconds，锁定期间：
    //   ① 右转决策锁住：再进 TrafficSignScene 直接复用右转，不发 VLM
    //   ② 跳过等宽：所有场景巡线都不做 normalizeTrackWidth（岔路新赛道不等宽）
    // ControlTask 主循环按真实帧间隔递减，到 0 自动解锁（恢复等宽 + 恢复判断）。
    // 全局生效：退出 TrafficSignScene 后仍锁定直到倒计时结束。
    std::atomic<float> right_turn_lock_seconds{0.0f};

    // ---- 分割→控制 唤醒机制 ----
    std::mutex              seg_cv_mtx_;
    std::condition_variable seg_cv_;

    // ---- 控制标记 ----
    std::atomic<bool> stop{false};

    // ---- 远程发车（ControlTask stdin 线程写，ControlTask 读，等价下位机按键） ----
    // 终端输入一行指令置位；consumeKeypress 与物理按键合并消费（发车/急停切换）
    std::atomic<bool> remote_keypress{false};
};
