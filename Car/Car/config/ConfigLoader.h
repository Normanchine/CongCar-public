#pragma once

#include "Config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

// ---- SceneControlParams 序列化（行为差异字段每场景各有） ----

inline void to_json(json& j, const SceneControlParams& s) {
    j = json{
        {"lateral_offset_ratio", s.lateral_offset_ratio},
        {"aim_dist_max", s.aim_dist_max},
        {"cx_smooth_frames", s.cx_smooth_frames},
        {"person_stop_ratio", s.person_stop_ratio},
    };
}

inline void from_json(const json& j, SceneControlParams& s) {
    if (j.contains("lateral_offset_ratio")) j.at("lateral_offset_ratio").get_to(s.lateral_offset_ratio);
    if (j.contains("aim_dist_max")) j.at("aim_dist_max").get_to(s.aim_dist_max);
    if (j.contains("cx_smooth_frames")) j.at("cx_smooth_frames").get_to(s.cx_smooth_frames);
    if (j.contains("person_stop_ratio")) j.at("person_stop_ratio").get_to(s.person_stop_ratio);
}

// ---- ForkConfig 序列化 ----

inline void to_json(json& j, const ForkConfig& f) {
    j = json{
        {"keep_side",       f.keep_side},
        {"confirm_rows",    f.confirm_rows},
        {"persist_frames",  f.persist_frames},
        {"width_threshold", f.width_threshold},
    };
}

inline void from_json(const json& j, ForkConfig& f) {
    if (j.contains("keep_side"))       j.at("keep_side").get_to(f.keep_side);
    if (j.contains("confirm_rows"))    j.at("confirm_rows").get_to(f.confirm_rows);
    if (j.contains("persist_frames"))  j.at("persist_frames").get_to(f.persist_frames);
    if (j.contains("width_threshold")) j.at("width_threshold").get_to(f.width_threshold);
}

// ---- ControlConfig 序列化 ----

inline void to_json(json& j, const ControlConfig& c) {
    j = json{
        {"max_speed",           c.max_speed},
        {"pid_p",               c.pid_p},
        {"pid_d",               c.pid_d},
        {"car_length",          c.car_length},
        {"car_cy_ratio",        c.car_cy_ratio},
        {"aim_dist_near",       c.aim_dist_near},
        {"pwm_mid",             c.pwm_mid},
        {"pwm_min",             c.pwm_min},
        {"pwm_max",             c.pwm_max},
        {"pwm_k",               c.pwm_k},
        {"steering_invert",        c.steering_invert},
        {"steer_speed_threshold", c.steer_speed_threshold},
        {"steer_speed_k",       c.steer_speed_k},
        {"lookahead_speed_k",   c.lookahead_speed_k},
        {"max_lost_frames",     c.max_lost_frames},
        {"coast_steer_step",   c.coast_steer_step},
        {"coast_speed",        c.coast_speed},
        {"speed_recovery_k",   c.speed_recovery_k},
        {"fork",                c.fork},
        {"scene",               c.scene}
    };
}

inline void from_json(const json& j, ControlConfig& c) {
    if (j.contains("max_speed")) j.at("max_speed").get_to(c.max_speed);
    if (j.contains("pid_p"))     j.at("pid_p").get_to(c.pid_p);
    if (j.contains("pid_d"))     j.at("pid_d").get_to(c.pid_d);
    j.at("car_length").get_to(c.car_length);
    j.at("car_cy_ratio").get_to(c.car_cy_ratio);
    if (j.contains("aim_dist_near"))    j.at("aim_dist_near").get_to(c.aim_dist_near);
    j.at("pwm_mid").get_to(c.pwm_mid);
    j.at("pwm_min").get_to(c.pwm_min);
    j.at("pwm_max").get_to(c.pwm_max);
    j.at("pwm_k").get_to(c.pwm_k);
    if (j.contains("steering_invert"))        j.at("steering_invert").get_to(c.steering_invert);
    if (j.contains("steer_speed_threshold"))  j.at("steer_speed_threshold").get_to(c.steer_speed_threshold);
    if (j.contains("steer_speed_k"))         j.at("steer_speed_k").get_to(c.steer_speed_k);
    if (j.contains("lookahead_speed_k"))   j.at("lookahead_speed_k").get_to(c.lookahead_speed_k);
    if (j.contains("max_lost_frames"))     j.at("max_lost_frames").get_to(c.max_lost_frames);
    if (j.contains("coast_steer_step"))    j.at("coast_steer_step").get_to(c.coast_steer_step);
    if (j.contains("coast_speed"))        j.at("coast_speed").get_to(c.coast_speed);
    if (j.contains("speed_recovery_k"))   j.at("speed_recovery_k").get_to(c.speed_recovery_k);
    if (j.contains("fork"))  j.at("fork").get_to(c.fork);
    if (j.contains("scene")) j.at("scene").get_to(c.scene);
}

// ---- SegmentationConfig 序列化 ----

inline void to_json(json& j, const SegmentationConfig& s) {
    j = json{
        {"model_path", s.model_path},
        {"input_w", s.input_w},
        {"input_h", s.input_h},
        {"min_region_px", s.min_region_px},
    };
}

inline void from_json(const json& j, SegmentationConfig& s) {
    j.at("model_path").get_to(s.model_path);
    j.at("input_w").get_to(s.input_w);
    j.at("input_h").get_to(s.input_h);
    if (j.contains("min_region_px")) j.at("min_region_px").get_to(s.min_region_px);
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

// ---- VlmConfig 序列化 ----

inline void to_json(json& j, const VlmConfig& v) {
    j = json{
        {"api_key",           v.api_key},
        {"base_url",          v.base_url},
        {"model",             v.model},
        {"deadline_seconds",  v.deadline_seconds},
        {"max_upload_width",  v.max_upload_width},
        {"max_upload_height", v.max_upload_height},
        {"box_padding_ratio", v.box_padding_ratio},
        {"stable_frames",     v.stable_frames},
        {"delay_frames",      v.delay_frames},
    };
}

inline void from_json(const json& j, VlmConfig& v) {
    if (j.contains("api_key"))           j.at("api_key").get_to(v.api_key);
    if (j.contains("base_url"))          j.at("base_url").get_to(v.base_url);
    if (j.contains("model"))             j.at("model").get_to(v.model);
    if (j.contains("deadline_seconds"))  j.at("deadline_seconds").get_to(v.deadline_seconds);
    if (j.contains("max_upload_width"))  j.at("max_upload_width").get_to(v.max_upload_width);
    if (j.contains("max_upload_height")) j.at("max_upload_height").get_to(v.max_upload_height);
    if (j.contains("box_padding_ratio")) j.at("box_padding_ratio").get_to(v.box_padding_ratio);
    if (j.contains("stable_frames"))     j.at("stable_frames").get_to(v.stable_frames);
    if (j.contains("delay_frames"))      j.at("delay_frames").get_to(v.delay_frames);
}

// ---- AIConfig 序列化（嵌套 segmentation + detection + vlm） ----

inline void to_json(json& j, const AIConfig& c) {
    j = json{
        {"segmentation", c.segmentation},
        {"detection",    c.detection},
        {"vlm",          c.vlm},
    };
}

inline void from_json(const json& j, AIConfig& c) {
    if (j.contains("segmentation")) j.at("segmentation").get_to(c.segmentation);
    if (j.contains("detection"))    j.at("detection").get_to(c.detection);
    if (j.contains("vlm"))          j.at("vlm").get_to(c.vlm);
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
