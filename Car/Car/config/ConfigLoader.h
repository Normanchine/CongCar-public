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
        {"car_length",          c.car_length},
        {"car_cy_ratio",        c.car_cy_ratio},
        {"aim_dist_near",       c.aim_dist_near},
        {"pwm_mid",             c.pwm_mid},
        {"pwm_min",             c.pwm_min},
        {"pwm_max",             c.pwm_max},
        {"pwm_k",               c.pwm_k},
        {"steering_invert",        c.steering_invert},
        {"steer_speed_threshold", c.steer_speed_threshold},
        {"lookahead_speed_k",   c.lookahead_speed_k},
        {"max_lost_frames",     c.max_lost_frames},
        {"coast_steer_step",   c.coast_steer_step},
        {"scene",               c.scene}
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
    if (j.contains("steering_invert"))        j.at("steering_invert").get_to(c.steering_invert);
    if (j.contains("steer_speed_threshold"))  j.at("steer_speed_threshold").get_to(c.steer_speed_threshold);
    if (j.contains("lookahead_speed_k"))   j.at("lookahead_speed_k").get_to(c.lookahead_speed_k);
    if (j.contains("max_lost_frames"))     j.at("max_lost_frames").get_to(c.max_lost_frames);
    if (j.contains("coast_steer_step"))    j.at("coast_steer_step").get_to(c.coast_steer_step);
    if (j.contains("scene")) j.at("scene").get_to(c.scene);
}

// ---- SegmentationConfig 序列化 ----

inline void to_json(json& j, const SegmentationConfig& s) {
    j = json{
        {"model_path", s.model_path},
        {"input_w", s.input_w},
        {"input_h", s.input_h},
    };
}

inline void from_json(const json& j, SegmentationConfig& s) {
    j.at("model_path").get_to(s.model_path);
    j.at("input_w").get_to(s.input_w);
    j.at("input_h").get_to(s.input_h);
}

// ---- DetectionConfig 序列化 ----

inline void to_json(json& j, const DetectionConfig& d) {
    j = json{
        {"model_path", d.model_path},
        {"input_w", d.input_w},
        {"input_h", d.input_h},
        {"confidence_threshold", d.confidence_threshold},
        {"nms_threshold", d.nms_threshold},
        {"lost_frame_limit", d.lost_frame_limit},
        {"class_num", d.class_num},
        {"classes", d.classes},
        {"colors", d.colors},
        {"scene_map", d.scene_map},
    };
}

inline void from_json(const json& j, DetectionConfig& d) {
    j.at("model_path").get_to(d.model_path);
    j.at("input_w").get_to(d.input_w);
    j.at("input_h").get_to(d.input_h);
    j.at("confidence_threshold").get_to(d.confidence_threshold);
    j.at("nms_threshold").get_to(d.nms_threshold);
    if (j.contains("lost_frame_limit")) j.at("lost_frame_limit").get_to(d.lost_frame_limit);
    if (j.contains("class_num"))  j.at("class_num").get_to(d.class_num);
    if (j.contains("classes"))    j.at("classes").get_to(d.classes);
    if (j.contains("colors"))     j.at("colors").get_to(d.colors);
    if (j.contains("scene_map"))  j.at("scene_map").get_to(d.scene_map);
}

// ---- AIConfig 序列化（嵌套 segmentation + detection） ----

inline void to_json(json& j, const AIConfig& c) {
    j = json{
        {"segmentation", c.segmentation},
        {"detection",    c.detection},
        {"ocr_model_path", c.ocr_model_path},
    };
}

inline void from_json(const json& j, AIConfig& c) {
    if (j.contains("segmentation")) j.at("segmentation").get_to(c.segmentation);
    if (j.contains("detection"))    j.at("detection").get_to(c.detection);
    if (j.contains("ocr_model_path")) j.at("ocr_model_path").get_to(c.ocr_model_path);
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UartConfig,
    port, baud, pwm_neutral)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(HttpStreamConfig,
    port, jpeg_quality, enable)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FrameCropConfig,
    enable, x, y, w, h)

inline void to_json(json& j, const CalibConfig& c) {
    j = json {
        {"src_quad",    {c.src_quad[0],c.src_quad[1],c.src_quad[2],c.src_quad[3],
                         c.src_quad[4],c.src_quad[5],c.src_quad[6],c.src_quad[7]}},
        {"bird_width",  c.bird_width},
        {"bird_height", c.bird_height},
        {"bird_width_scale", c.bird_width_scale},
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
    if (j.contains("bird_width_scale")) j.at("bird_width_scale").get_to(c.bird_width_scale);
    j.at("ppm_x").get_to(c.ppm_x);
    j.at("ppm_y").get_to(c.ppm_y);
}

inline void to_json(json& j, const AppConfig& c) {
    j = json {
        {"debug_mode",  c.debug_mode},
        {"shm_name",    c.shm_name},
        {"shm_size",    c.shm_size},
        {"frame_crop",  c.frame_crop},
        {"control",     c.control},
        {"ai",          c.ai},
        {"calib",       c.calib},
        {"uart",        c.uart},
        {"http_stream", c.http_stream}
    };
}

inline void from_json(const json& j, AppConfig& c) {
    j.at("debug_mode").get_to(c.debug_mode);
    j.at("shm_name").get_to(c.shm_name);
    j.at("shm_size").get_to(c.shm_size);
    if (j.contains("frame_crop")) j.at("frame_crop").get_to(c.frame_crop);
    j.at("control").get_to(c.control);
    if (j.contains("ai"))     j.at("ai").get_to(c.ai);
    if (j.contains("calib"))  j.at("calib").get_to(c.calib);
    if (j.contains("uart"))   j.at("uart").get_to(c.uart);
    if (j.contains("http_stream")) j.at("http_stream").get_to(c.http_stream);
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
