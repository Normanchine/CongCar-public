#pragma once

#include "Config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

// ---- SceneControlParams 序列化 ----

inline void to_json(json& j, const SceneControlParams& s) {
    j = json{ {"max_speed", s.max_speed}, {"pid_p", s.pid_p}, {"pid_d", s.pid_d} };
}

inline void from_json(const json& j, SceneControlParams& s) {
    j.at("max_speed").get_to(s.max_speed);
    j.at("pid_p").get_to(s.pid_p);
    j.at("pid_d").get_to(s.pid_d);
}

// ---- ControlConfig 序列化 ----

inline void to_json(json& j, const ControlConfig& c) {
    j = json{
        {"car_length",    c.car_length},
        {"car_cy_ratio",  c.car_cy_ratio},
        {"aim_dist_near", c.aim_dist_near},
        {"pwm_mid",       c.pwm_mid},
        {"pwm_min",       c.pwm_min},
        {"pwm_max",       c.pwm_max},
        {"pwm_k",         c.pwm_k},
        {"scene",         c.scene}
    };
}

inline void from_json(const json& j, ControlConfig& c) {
    j.at("car_length").get_to(c.car_length);
    j.at("car_cy_ratio").get_to(c.car_cy_ratio);
    j.at("aim_dist_near").get_to(c.aim_dist_near);
    j.at("pwm_mid").get_to(c.pwm_mid);
    j.at("pwm_min").get_to(c.pwm_min);
    j.at("pwm_max").get_to(c.pwm_max);
    j.at("pwm_k").get_to(c.pwm_k);
    if (j.contains("scene")) j.at("scene").get_to(c.scene);
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AIConfig,
    confidence_threshold, nms_threshold,
    detection_model_path, segmentation_model_path, ocr_model_path,
    segmentation_input_w, segmentation_input_h,
    detection_input_w, detection_input_h)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UartConfig,
    port, baud, pwm_neutral)

inline void to_json(json& j, const CalibConfig& c) {
    j = json {
        {"src_quad",    {c.src_quad[0],c.src_quad[1],c.src_quad[2],c.src_quad[3],
                         c.src_quad[4],c.src_quad[5],c.src_quad[6],c.src_quad[7]}},
        {"bird_width",  c.bird_width},
        {"bird_height", c.bird_height},
        {"ppm_x",       c.ppm_x},
        {"ppm_y",       c.ppm_y}
    };
}

inline void from_json(const json& j, CalibConfig& c) {
    auto arr = j.at("src_quad").get<std::vector<float>>();
    for (size_t i = 0; i < std::min(arr.size(), size_t(8)); ++i)
        c.src_quad[i] = arr[i];
    j.at("bird_width").get_to(c.bird_width);
    j.at("bird_height").get_to(c.bird_height);
    j.at("ppm_x").get_to(c.ppm_x);
    j.at("ppm_y").get_to(c.ppm_y);
}

inline void to_json(json& j, const AppConfig& c) {
    j = json {
        {"debug_mode", c.debug_mode},
        {"shm_name",   c.shm_name},
        {"shm_size",   c.shm_size},
        {"control",    c.control},
        {"ai",         c.ai},
        {"calib",      c.calib},
        {"uart",       c.uart}
    };
}

inline void from_json(const json& j, AppConfig& c) {
    j.at("debug_mode").get_to(c.debug_mode);
    j.at("shm_name").get_to(c.shm_name);
    j.at("shm_size").get_to(c.shm_size);
    j.at("control").get_to(c.control);
    if (j.contains("ai"))     j.at("ai").get_to(c.ai);
    if (j.contains("calib"))  j.at("calib").get_to(c.calib);
    if (j.contains("uart"))   j.at("uart").get_to(c.uart);
}

// ---- 加载函数 ----

inline bool loadConfig(const std::string& path, AppConfig& config) {
    try {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[Config] 无法打开配置文件: " << path << std::endl;
            return false;
        }
        json j = json::parse(f);
        from_json(j, config);
        std::cout << "[Config] 已加载: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Config] JSON 解析失败: " << e.what() << std::endl;
        return false;
    }
}
