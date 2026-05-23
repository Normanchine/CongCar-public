#pragma once

#include <string>

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

    // ---- PID ----
    float max_speed = 0.0f;            // 车速限制
    float pid_p = 0.0f;
    float pid_i = 0.0f;
    float pid_d = 0.0f;
};

struct AIConfig {
    float confidence_threshold = 0.5f;       // 置信度阈值
    float nms_threshold        = 0.45f;      // NMS 阈值

    std::string detection_model_path =        // 目标检测模型
        "/home/orangepi/Desktop/setupUI-1.0.6_app-1.0.6/setupUI/"
        "infer_wrap/base/model/rknn_lt.rknn";
    std::string segmentation_model_path = 
        "/home/orangepi/Desktop/seg_python/model/pplite_seg.rknn";     // 语义分割模型
    std::string ocr_model_path;              // OCR 模型

    int segmentation_input_w = 160;
    int segmentation_input_h = 160;
    int detection_input_w    = 640;
    int detection_input_h    = 640;
};

struct CalibConfig {
    float src_quad[8] = {0};           // 透视源四点
    int   bird_width  = 640;
    int   bird_height = 480;
    float ppm         = 222.22f;       // 鸟瞰像素/米
};

struct UartConfig {
    std::string port        = "/dev/ttyUSB0";
    int         baud        = 115200;
    float       pwm_neutral = 1500.0f;
};

struct AppConfig {
    bool        debug_mode = true;
    std::string shm_name   = "shm_ar_video";
    size_t      shm_size   = 10 * 1024 * 1024;  // 10MB

    ControlConfig control;
    AIConfig      ai;
    CalibConfig   calib;
    UartConfig    uart;
};
