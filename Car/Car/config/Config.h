#pragma once

#include <map>
#include <string>
#include <vector>

// TrafficSignScene 右转子配置（嵌套在 SceneControlParams::right）
struct TurnControlParams {
    float max_speed = 0.55f;
    float pid_p     = 0.9f;
    float pid_d     = 0.7f;
};

struct SceneControlParams {
    float max_speed = 1.0f;
    float pid_p     = 0.0f;
    float pid_d     = 0.0f;
    float aim_dist_near    = 0.35f;  // 近前视距离 (m)
    float coast_speed_decay = 0.9f;  // 盲开速度衰减系数
    float speed_recovery_k  = 0.3f;  // 加速恢复系数

    // TrafficSignScene 右转专属参数
    TurnControlParams right;

};

struct ForkConfig {
    int   keep_side      = 0;     // 保留侧：0=左，1=右
    int   confirm_rows   = 3;     // 连续分叉行数阈值（确认岔路）
    int   persist_frames = 30;    // 岔路持久化帧数
    float width_threshold = 1.3f; // 持久化模式宽度阈值（行宽>stem_w*ratio才切）
};

struct ControlConfig {
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
    float coast_speed_decay = 0.9f;    // 盲开速度衰减系数（每帧乘以此值）
    float speed_recovery_k  = 0.3f;    // 加速恢复系数（0-1，越大恢复越快，1=瞬时）

    // ---- 岔路策略 ----
    ForkConfig fork;

    // ---- 场景参数 ----
    std::map<std::string, SceneControlParams> scene;
};

struct SegmentationConfig {
    std::string model_path =
        "/home/orangepi/Desktop/seg_python/model/pplite_seg.rknn";
    int input_w = 160;
    int input_h = 160;
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
