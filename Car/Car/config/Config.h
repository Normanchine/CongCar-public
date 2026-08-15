#pragma once

#include <map>
#include <string>
#include <vector>

// 场景行为参数：每场景各有（贴边偏移量 / 前视上限 / cx平滑）
struct SceneControlParams {
    // 沿边参考点横向偏移（鸟瞰宽比例，默认0=压线）
    // 贴边界内侧：左边界→偏右(cx增大)；右边界→偏左(cx减小)
    float lateral_offset_ratio = 0.0f;

    // 前视点最大距离 (m)，0=不限制。CarScene 沿边界走时限制前视，避免远处误差放大
    float aim_dist_max = 0.0f;

    // PersonScene：行人底部到达鸟瞰图高度的此比例时停车（0~1，默认0.5=鸟瞰图一半高度）
    float person_stop_ratio = 0.5f;

    // TrafficSignScene：右转决策后盲开帧数（递减渐变右转，走够后恢复巡线）
    int   turn_blind_frames = 30;
    float turn_blind_speed  = 0.5f;   // 右转盲开固定速度 (m/s)

    // TrafficSignScene：停车后倒车帧数（停车受惯性冲过头，倒车退回一段距离）
    int   backup_frames = 20;
    float backup_speed  = -0.1f;   // 倒车速度 (m/s)，负值=后退

    // TrafficSignScene：右转决策后的全局右转锁时长（秒）。
    // 右转后锁定：① 再进场景直接复用右转不发VLM ② 全局跳过等宽。ControlTask 每帧递减，到 0 解锁。
    float right_turn_lock_seconds = 3.0f;
};

struct ForkConfig {
    int   keep_side      = 0;     // 保留侧：0=左，1=右
    int   confirm_rows   = 3;     // 连续分叉行数阈值（确认岔路）
    int   persist_frames = 30;    // 岔路持久化帧数
    float width_threshold = 1.3f; // 持久化模式宽度阈值（行宽>stem_w*ratio才切）
};

struct ControlConfig {
    // ---- 全局控制参数（所有场景统一，切换场景无参数跳变） ----
    float max_speed = 1.0f;            // 目标速度 (m/s)
    float pid_p     = 0.0f;            // P 系数（左边界）
    float pid_d     = 0.0f;            // D 系数（左边界）
    float pid_p_right = 0.0f;          // P 系数（右边界，0=回退用左边界）
    float pid_d_right = 0.0f;          // D 系数（右边界，0=回退用左边界）

    // 全局沿边方向初始值：-1=左边界 +1=右边界 0=中点。
    // 各场景首帧从此继承（不再固定左边界）；CarScene 每帧按障碍位置改写
    int default_follow_edge = -1;

    // ---- 物理参数 ----
    float car_length    = 0.316f;      // 轴距 (m)
    float car_cy_ratio  = 0.95f;       // 车底中心 y = 车高 * ratio
    float aim_dist_near = 0.35f;       // 近前视距离 (m)

    // ---- PWM ----
    float pwm_mid = 1500.0f;           // PWM 中值
    float pwm_min = 500.0f;            // PWM 下限
    float pwm_max = 2500.0f;           // PWM 上限
    float pwm_k   = 30.0f;             // 误差 → PWM 缩放
    bool  steering_invert = false;     // AR图像左右镜像时反转转向
    float steer_speed_threshold = 0.15f;  // |error|超过此值(rad)启动降速
    float steer_speed_k         = 2.0f;   // 弯道降速系数：target = max_speed / (1 + k * excess)

    // ---- 动态前视 ----
    float lookahead_speed_k = 0.15f;   // 动态前视系数: aim = near + speed * k

    // ---- 丢线容错 ----
    int   max_lost_frames   = 5;       // 最大盲开帧数
    float coast_steer_step  = 200.0f;  // 盲开最大搜转偏移量，逐帧递增(PWM)
    float coast_speed       = 0.5f;    // 盲开固定速度 (m/s)，丢线后按此速度保持
    float speed_recovery_k  = 0.3f;    // 加速恢复系数（0-1，越大恢复越快，1=瞬时）

    // ---- 岔路策略 ----
    ForkConfig fork;

    // ---- 场景参数（lateral_offset_ratio 每场景各有） ----
    std::map<std::string, SceneControlParams> scene;
};

struct SegmentationConfig {
    std::string model_path =
        "/home/orangepi/Desktop/seg_python/model/pplite_seg.rknn";
    int input_w = 160;
    int input_h = 160;

    // 后处理去噪：连通域长宽任一小于此像素数 → 判定为干扰碎片，剪掉。
    // 在推理分辨率(seg_w×seg_h)上判定，0=不过滤
    int min_region_px = 10;
};

struct DetectionConfig {
    std::string model_path =
        "/home/orangepi/Desktop/setupUI-1.0.6_app-1.0.6/setupUI/"
        "infer_wrap/base/model/rknn_lt.rknn";
    int input_w = 320;
    int input_h = 256;
    float confidence_threshold = 0.40f;
    float nms_threshold        = 0.65f;
    int   lost_frame_limit     = 5;    // 特殊场景切回NormalScene的缓冲帧数

    // 类别配置
    int class_num = 0;
    std::vector<std::string> classes;
    std::vector<std::vector<int>> colors;
    std::map<std::string, std::string> scene_map;
};

struct VlmConfig {
    std::string api_key          = "";
    std::string base_url         = "https://qianfan.baidubce.com/v2";
    std::string model            = "ernie-4.5-turbo-vl-32k";
    float       deadline_seconds = 4.0f;
    int         max_upload_width  = 250;
    int         max_upload_height = 96;
    float       box_padding_ratio = 0.06f;
    int         stable_frames     = 3;
    int         delay_frames      = 0;    // 稳定后延迟帧数（等车靠近标志）
    int         max_tokens        = 400;  // 输出 token 上限（思考+动作）
};

struct AIConfig {
    DetectionConfig    detection;
    SegmentationConfig segmentation;
    VlmConfig          vlm;
};

struct CalibConfig {
    float src_quad[8] = {0};           // 透视源四点
    int   bird_width  = 640;
    int   bird_height = 480;
    float bird_width_scale = 1.0f;     // 鸟瞰图宽度扩展倍数（>1扩大视野，ppm不变）
    float ppm_x       = 0.0f;          // X方向像素/米
    float ppm_y       = 0.0f;          // Y方向像素/米
};

struct UartConfig {
    std::string port        = "/dev/ttyUSB0";
    int         baud        = 115200;
    float       pwm_neutral = 1500.0f;
};

struct HttpStreamConfig {
    int  port         = 5001;     // HTTP服务器端口
    int  jpeg_quality = 80;       // JPEG编码质量（1-100）
    bool enable       = true;     // 是否启用HTTP推流
};

struct FrameCropConfig {
    bool enable = false;
    int  x      = 0;
    int  y      = 240;
    int  w      = 640;
    int  h      = 240;
};

struct AppConfig {
    bool        debug_mode = true;
    std::string shm_name   = "shm_ar_video";
    size_t      shm_size   = 10 * 1024 * 1024;  // 10MB

    FrameCropConfig frame_crop;
    ControlConfig   control;
    AIConfig        ai;
    CalibConfig     calib;
    UartConfig      uart;
    HttpStreamConfig http_stream;
};
