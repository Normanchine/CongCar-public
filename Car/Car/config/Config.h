#pragma once

#include <string>
#include <map>

struct SceneControlParams {
    float max_speed = 1.0f;
    float pid_p     = 0.0f;
    float pid_d     = 0.0f;
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

    // ---- 动态前视 ----
    float lookahead_speed_k = 0.15f;   // 动态前视系数: aim = near + speed * k

    // ---- 丢线容错 ----
    int   max_lost_frames   = 5;       // 最大盲开帧数
    float coast_steer_step  = 200.0f;  // 盲开每帧舵机偏移量(PWM)

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
    int input_w = 640;
    int input_h = 640;
    float confidence_threshold = 0.5f;
    float nms_threshold        = 0.45f;
    int   lost_frame_limit     = 5;    // 特殊场景切回NormalScene的缓冲帧数

    // 类别配置
    int class_num = 0;
    std::vector<std::string> classes;
    std::vector<std::vector<int>> colors;
    std::map<std::string, std::string> scene_map;
};

struct AIConfig {
    DetectionConfig    detection;
    SegmentationConfig segmentation;
    std::string        ocr_model_path;
};

struct CalibConfig {
    float src_quad[8] = {0};           // 透视源四点
    int   bird_width  = 640;
    int   bird_height = 480;
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
