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

    // ---- 控制输出 ----
    std::atomic<float> steering_pwm{1500};   // 舵机PWM (500-2500, 中值1500)
    std::atomic<float> target_speed{0};      // 目标速度
    float steering_dir = 1.0f;               // 转向方向（AR图像左右镜像时置-1）

    // ---- 透视变换 ----
    PerspectiveMapper perspective;

    // ---- 原始帧尺寸（CaptureTask 写入，DetectionTask 读取用于坐标映射） ----
    std::atomic<int> frame_width{0};
    std::atomic<int> frame_height{0};

    // ---- 分割→控制 唤醒机制 ----
    std::mutex              seg_cv_mtx_;
    std::condition_variable seg_cv_;

    // ---- 控制标记 ----
    std::atomic<bool> stop{false};
};
