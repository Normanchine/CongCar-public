#include "TrafficSignScene.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include "../../common/Base64.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

REGISTER_SCENE(TrafficSignScene)

using json = nlohmann::json;

// ---- VLM 系统提示词（与 Python 原型一致） ----
static const char* kVlmSystemPrompt = R"PROMPT(你是智能车固定岔路口的快速视觉决策模块。

输入是一张由 YOLO 裁剪得到的中文路口指示牌图片。

车辆只有两个动作：
直走：继续沿外侧大圈逆时针行驶。
右转：在当前岔路口进入中间小圈。

根据指示牌文字的最终真实含义，只输出一个动作。

右转必须存在明确依据，只有以下情况才能输出"右转"：
1. 指示牌明确要求、推荐或引导走右侧、右转、进入内圈、走近道；
2. 指示牌明确表示直道不能走、直行禁止、直道封闭或继续直行危险。

其他所有情况都输出"直走"。

必须理解完整句意，不能按单个关键词判断。
必须正确理解否定、反问、转折、强调和调侃：
- "才怪"会否定前面紧邻的说法；
- "不是、不能、不许、不要、并不"等表示否定；
- "但是、不过、然而、却"等之后的最终结论优先；
- 仅出现"右道、右侧、右转"等词，但没有明确推荐右侧，不构成右转依据；
- 右道条件较差但直道明确无法通行时，仍输出"右转"；
- 图片模糊、文字无法识别、语义不明确或信息无关时，输出"直走"。

忽略背景、边框、装饰和其他无关内容。
不要解释，不要复述，不要输出标点或 JSON。

最终只能输出：
直走
或
右转)PROMPT";

// ---- 辅助：libcurl 写回调 ----
namespace {

struct CurlBuffer {
    std::string data;
};

size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* buf = static_cast<CurlBuffer*>(userp);
    size_t total = size * nmemb;
    buf->data.append(static_cast<char*>(contents), total);
    return total;
}

} // anonymous namespace

// ---- 重写 onFork：岔路口锁舵降速 ----
void TrafficSignScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    const auto& fk = ctx->config.control.fork;

    // 保存原始 mask（锁舵期间需恢复，撤销切除）
    cv::Mat original;
    if (fork_lock_ || fork_state_.persist_remaining == 0) {
        original = wide_mask.clone();
    }

    // 调用 handleFork 进行分叉检测
    pathplan::handleFork(wide_mask, fork_state_,
                         fork_side, fk.confirm_rows,
                         fk.persist_frames, fk.width_threshold, 1);

    // 第一帧进入 persist → 锁定转向
    if (!fork_lock_ && fork_state_.persist_remaining > 0) {
        fork_lock_ = true;
        fork_pwm_  = last_pwm_;   // 进入分叉口的自然入弯 PWM
    }

    // 锁舵期间：恢复原始 mask，不切赛道
    if (fork_lock_ && fork_state_.persist_remaining > 0) {
        original.copyTo(wide_mask);
    }

    // persist 结束 → 解锁，后续帧正常切赛道
    if (fork_lock_ && fork_state_.persist_remaining == 0) {
        fork_lock_ = false;
    }
}

// ---- 重写 onPD：锁舵期间返回冻结 PWM ----
float TrafficSignScene::Follower::onPD(float error, SharedContext* ctx) {
    if (fork_lock_) {
        last_error_ = 0.0f;  // 清零防止解锁后 D 项跳变
        return fork_pwm_;
    }
    return PathFollower::onPD(error, ctx);
}

// ---- 重写 onSpeed：锁舵期间使用岔路口恒定低速 ----
float TrafficSignScene::Follower::onSpeed(float error, SharedContext* ctx) {
    if (fork_lock_) {
        const auto& sp = ctx->config.control.scene.at("TrafficSignScene");
        return sp.fork_speed;
    }
    return PathFollower::onSpeed(error, ctx);
}

// ---- 清理 VLM 输出文本 ----
std::string TrafficSignScene::cleanOutput(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char ch : text) {
        // 过滤 ASCII 空白和控制字符
        if (ch == '\r' || ch == '\n' || ch == '\t' || ch == ' ') continue;
        // 过滤 ASCII 标点
        if (ch == ',' || ch == '.' || ch == '!' || ch == '?' ||
            ch == ':' || ch == ';' || ch == '"' || ch == '\'' || ch == '`') continue;
        result.push_back(ch);
    }
    return result;
}

// ---- 解析 VLM 输出（0=直走，1=右转，-1=非法） ----
static int parseAction(const std::string& cleaned) {
    if (cleaned == "直走") return 0;
    if (cleaned == "右转") return 1;
    return -1;
}

// ---- 解析 early action（前缀匹配，兼容流式截断输出） ----
static int parseEarlyAction(const std::string& cleaned) {
    if (cleaned.compare(0, strlen("直走"), "直走") == 0) return 0;
    if (cleaned.compare(0, strlen("右转"), "右转") == 0) return 1;
    return -1;
}

// ---- 准备发送给 VLM 的图片 ----
std::vector<uint8_t> TrafficSignScene::prepareSignImage(const cv::Mat& frame,
                                                         float x1, float y1,
                                                         float x2, float y2,
                                                         const VlmConfig& cfg) {
    int w = frame.cols, h = frame.rows;
    float box_w = x2 - x1;
    float box_h = y2 - y1;
    float pad_x = box_w * cfg.box_padding_ratio;
    float pad_y = box_h * cfg.box_padding_ratio;

    int cx1 = std::max(0,         static_cast<int>(x1 - pad_x));
    int cy1 = std::max(0,         static_cast<int>(y1 - pad_y));
    int cx2 = std::min(w,         static_cast<int>(x2 + pad_x));
    int cy2 = std::min(h,         static_cast<int>(y2 + pad_y));

    if (cx2 <= cx1 || cy2 <= cy1) {
        cx1 = std::max(0, static_cast<int>(x1));
        cy1 = std::max(0, static_cast<int>(y1));
        cx2 = std::min(w, static_cast<int>(x2));
        cy2 = std::min(h, static_cast<int>(y2));
    }

    cv::Mat crop = frame(cv::Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
    cv::Mat resized;
    cv::resize(crop, resized,
               cv::Size(cfg.max_upload_width, cfg.max_upload_height),
               0, 0, cv::INTER_LANCZOS4);

    std::vector<uint8_t> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    cv::imencode(".jpg", resized, jpeg, params);
    return jpeg;
}

// ---- 解析 API Key ----
static std::string resolveApiKey(const VlmConfig& cfg) {
    return cfg.api_key;
}

// ---- 同步调用千帆 VLM API（在后台线程中执行） ----
int TrafficSignScene::callVlmApi(const std::vector<uint8_t>& jpeg_bytes,
                                  const VlmConfig& cfg) {
    std::string api_key = resolveApiKey(cfg);
    if (api_key.empty())
        throw std::runtime_error(
            "未配置 VLM API Key（config.vlm.api_key）");

    // 1. Base64 编码 → data URL
    std::string b64 = base64::encode(jpeg_bytes);
    std::string image_url = "data:image/jpeg;base64," + b64;

    // 2. 构造请求体
    json request_body = {
        {"model",       cfg.model},
        {"temperature", 0},
        {"max_tokens",  4},
        {"stream",      false},
        {"messages", json::array({
            {{"role", "system", "content", kVlmSystemPrompt}},
            {{"role", "user",   "content", json::array({
                {{"type", "image_url", "image_url", {{"url", image_url}, {"detail", "low"}}}},
                {{"type", "text",       "text", "只输出最终动作。"}}
            })}}
        })}
    };
    std::string body_str = request_body.dump();

    // 3. libcurl 配置
    CURL* curl = curl_easy_init();
    if (!curl)
        throw std::runtime_error("curl_easy_init 失败");

    CurlBuffer response_buf;
    std::string url = cfg.base_url + "/chat/completions";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_header = "Authorization: Bearer " + api_key;
    headers = curl_slist_append(headers, auth_header.c_str());

    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,     body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,  body_str.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &response_buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        static_cast<long>(cfg.deadline_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(
            std::string("千帆 API 请求失败: ") + curl_easy_strerror(res));
    }

    if (http_code != 200) {
        throw std::runtime_error(
            "千帆 API 返回 HTTP " + std::to_string(http_code) + ": " + response_buf.data);
    }

    // 4. 解析响应
    json resp = json::parse(response_buf.data);
    std::string content = resp["choices"][0]["message"]["content"].get<std::string>();

    // 5. 清理 + 匹配动作
    std::string cleaned = cleanOutput(content);
    int action = parseAction(cleaned);
    if (action < 0) {
        action = parseEarlyAction(cleaned);
    }
    if (action < 0) {
        std::cerr << "[TrafficSignScene] VLM 输出无法解析: \""
                  << content << "\" → 清洗后: \"" << cleaned
                  << "\"，兜底直走" << std::endl;
        return 0;
    }

    return action;
}

// ---- 主循环 ----
void TrafficSignScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    const auto& vlm_cfg = ctx->config.ai.vlm;
    const auto& det_cfg = ctx->config.ai.detection;

    // ---- 查找 traffic_sign class_id（只查一次） ----
    if (traffic_sign_class_id_ < 0) {
        const auto& classes = det_cfg.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "traffic_sign") {
                traffic_sign_class_id_ = i;
                break;
            }
        }
    }

    // ---- 获取最新检测结果 ----
    auto det = ctx->det_result.get();

    // ---- VLM 状态机 ----
    switch (vlm_state_) {

    case VlmState::IDLE: {
        const DetectionBox* best = nullptr;
        float best_score = 0.0f;
        if (det && traffic_sign_class_id_ >= 0) {
            for (const auto& b : det->boxes) {
                if (b.class_id == traffic_sign_class_id_ && b.confidence > best_score) {
                    best_score = b.confidence;
                    best = &b;
                }
            }
        }

        if (best && best_score > 0.4f) {
            stable_count_++;
            if (stable_count_ >= vlm_cfg.stable_frames) {
                cv::Mat work_frame;
                if (ctx->work_frame_queue.tryConsume(work_frame)) {
                    auto jpeg_bytes = prepareSignImage(
                        work_frame, best->x1, best->y1, best->x2, best->y2, vlm_cfg);

                    std::cout << "[TrafficSignScene] 稳定检测到交通标志 (score="
                              << best_score << ")，发起 VLM 请求..."
                              << " 图片大小=" << jpeg_bytes.size() / 1024 << "KB"
                              << std::endl;

                    // detached thread + shared_ptr<atomic>：避免 future 析构阻塞
                    vlm_result_ = std::make_shared<std::atomic<int>>(-1);
                    std::thread([jpeg = std::move(jpeg_bytes),
                                 &cfg = vlm_cfg,
                                 result = vlm_result_]() {
                        try {
                            int action = callVlmApi(jpeg, cfg);
                            result->store(action, std::memory_order_release);
                        } catch (const std::exception& e) {
                            std::cerr << "[TrafficSignScene] VLM 请求异常: "
                                      << e.what() << "，兜底直走" << std::endl;
                            result->store(0, std::memory_order_release);
                        }
                    }).detach();
                    vlm_state_ = VlmState::REQUESTING;
                } else {
                    std::cerr << "[TrafficSignScene] 无可用工作帧，跳过 VLM 请求"
                              << std::endl;
                }
                stable_count_ = 0;
            }
        } else {
            stable_count_ = 0;
        }
        break;
    }

    case VlmState::REQUESTING: {
        if (vlm_result_) {
            int r = vlm_result_->load(std::memory_order_acquire);
            if (r >= 0) {
                cached_fork_side_ = r;
                vlm_state_ = VlmState::DECIDED;
                std::cout << "[TrafficSignScene] VLM 决策完成: "
                          << (cached_fork_side_ == 0 ? "直走" : "右转") << std::endl;
            }
        }
        break;
    }

    case VlmState::DECIDED:
        break;
    }

    // ---- 设置分岔方向并巡线 ----
    follower_.fork_side = cached_fork_side_;
    follower_.process(seg->mask, ctx);
}
