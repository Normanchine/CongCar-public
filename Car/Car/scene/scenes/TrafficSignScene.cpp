#include "TrafficSignScene.h"
#include "../../hal/uart/UartIO.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include "../planning/PathPlanning.h"
#include "../../common/Base64.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

REGISTER_SCENE(TrafficSignScene)

using json = nlohmann::json;

// ---- VLM 系统提示词 ----
static const char* kVlmSystemPrompt = R"PROMPT(你是智能车唯一一个固定岔路口的路线裁判。

输入是一张中文路口指示牌图片。这里只有两条可选道路，必须二选一：
- 直走：继续沿外侧大圈逆时针行驶；
- 右转：进入右侧岔路，也就是中间小圈。

你的任务是读完整句意，判断哪一条路还能走。不要因为某条路"条件差"就把它当成"不能走"。

严格按以下优先级决策：
1. 只要牌子明确说直道、直行、前方、外圈"不能走、走不了、禁止、封闭、不通、危险、断路"，且没有明确说右道也不能走，立刻输出"右转"。这是最高优先级。
2. 只要牌子明确要求、推荐或允许走右道、右侧、右转、内圈、近道、捷径、抄近道、短路线，输出"右转"。
3. 只有牌子明确说右道、右转、内圈"不能走、禁止、封闭、不通、危险"，而直道仍可走时，输出"直走"。
4. 两条路都没有明确限制或推荐时，输出"直走"。

必须正确理解转折和否定：
- "虽然 A，但是 B"中，A 只是让步条件，B 才是最终有效结论；
- "右道崎岖、窄、不好走、绕、远、颠"只表示右道体验差，不等于右道不能走；
- "右道虽然崎岖，但直道真的不能走"必须输出"右转"；
- "右道不好走，但是直道封闭"必须输出"右转"；
- "右道不能走，直道可以通行"必须输出"直走"；
- "直道才怪不能走"表示"直道能走"，不能因"不能走"三个字而右转；
- 否定只作用于它所否定的道路，不能把"右道崎岖"误判为"右道禁止"。

忽略背景、边框和无关文字。图片模糊、文字无法识别或确实没有路线信息时才默认输出"直走"。
不要解释、不要复述、不要输出标点或 JSON。
最终只能输出"直走"或"右转"。")PROMPT";

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

// ---- 重写 onPD：按 fork_side 选直走/右转两套 PID ----
float TrafficSignScene::Follower::onPD(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp   = ctrl.scene.at("TrafficSignScene");
    float pp = (fork_side == 1) ? sp.right.pid_p : sp.pid_p;
    float pd = (fork_side == 1) ? sp.right.pid_d : sp.pid_d;

    float derivative = first_frame_ ? 0.0f : (error - last_error_);
    last_error_ = error;

    float output = pp * error + pd * derivative;
    float pwm = ctrl.pwm_mid - output * ctrl.pwm_k;
    return std::clamp(pwm, ctrl.pwm_min, ctrl.pwm_max);
}

// ---- 重写 onSpeed：按 fork_side 选直走/右转两套 max_speed ----
float TrafficSignScene::Follower::onSpeed(float error, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;
    const auto& sp   = ctrl.scene.at("TrafficSignScene");
    float ms = (fork_side == 1) ? sp.right.max_speed : sp.max_speed;

    float target = ms;
    float ae = std::abs(error);
    if (ae > ctrl.steer_speed_threshold) {
        float excess = ae - ctrl.steer_speed_threshold;
        target = ms / (1.0f + ctrl.steer_speed_k * excess);
    }

    // 减速立即生效，加速平滑恢复
    if (target < last_speed_) {
        return target;
    }
    return last_speed_ + (target - last_speed_) * ctrl.speed_recovery_k;
}

// ---- 清理 VLM 输出文本（与 Python 一致：移除空白+中英文标点） ----
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
    // 过滤中文标点（UTF-8 多字节序列）
    static const std::string kCnPunct[] = {
        "\xEF\xBC\x8C",  // ，
        "\xE3\x80\x82",  // 。
        "\xEF\xBC\x81",  // ！
        "\xEF\xBC\x9F",  // ？
        "\xEF\xBC\x9A",  // ：
        "\xEF\xBC\x9B",  // ；
    };
    for (const auto& p : kCnPunct) {
        size_t pos;
        while ((pos = result.find(p)) != std::string::npos) {
            result.erase(pos, p.length());
        }
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

    // 不缩放，保持原始分辨率直接 JPEG 编码，避免文字压缩模糊
    std::vector<uint8_t> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 92};
    cv::imencode(".jpg", crop, jpeg, params);
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

    std::cout << "[TrafficSignScene] VLM 请求: url=" << (cfg.base_url + "/chat/completions")
              << " model=" << cfg.model
              << " deadline=" << cfg.deadline_seconds << "s"
              << " key_prefix=" << api_key.substr(0, 12) << "..." << std::endl;

    // 1. Base64 编码 → data URL
    std::string b64 = base64::encode(jpeg_bytes);
    std::string image_url = "data:image/jpeg;base64," + b64;

    // 2. 构造请求体（stream=true，与 Python 一致）。
    // 注意：nlohmann::json 的 {"key", value, "key", value} 会被序列化为数组，
    // 不能省略每个键值对外层的 {}。此前 messages 实际发送成了二维数组，
    // 因而被千帆拒绝为 malformed_json。
    json user_content = json::array({
        {
            {"type", "image_url"},
            {"image_url", {
                {"url", image_url},
                {"detail", "low"},
            }},
        },
        {
            {"type", "text"},
            {"text", "只输出最终动作。注意先读完整句：近道、捷径、更短路线类表达通常指右转，但被否定或被\"才怪\"等推翻时不能按右转处理。"},
        },
    });
    json messages = json::array({
        {
            {"role", "system"},
            {"content", kVlmSystemPrompt},
        },
        {
            {"role", "user"},
            {"content", std::move(user_content)},
        },
    });
    json request_body = {
        {"model",       cfg.model},
        {"temperature", 0},
        {"max_tokens",  4},
        {"stream",      true},
        {"messages",    std::move(messages)},
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

    // 4. 解析 SSE 流式响应（与 Python 一致：逐 chunk 拼内容，早期匹配）
    std::string raw_output;
    {
        std::istringstream stream(response_buf.data);
        std::string line;
        while (std::getline(stream, line)) {
            // SSE 格式: "data: <json>" 或 "data: [DONE]"
            if (line.compare(0, 6, "data: ") != 0) continue;
            std::string payload = line.substr(6);
            if (payload == "[DONE]") break;

            try {
                auto chunk = json::parse(payload);
                if (chunk.contains("choices") && !chunk["choices"].empty()) {
                    auto& delta = chunk["choices"][0]["delta"];
                    if (delta.contains("content")) {
                        raw_output += delta["content"].get<std::string>();
                    }
                }
            } catch (const json::parse_error&) {
                // 忽略无法解析的行
            }
        }
    }

    // 5. 清理 + 匹配动作
    std::string cleaned = cleanOutput(raw_output);

    // 始终打印 VLM 原始输出，方便排查识别错误
    std::cout << "[TrafficSignScene] VLM 原始输出: \"" << raw_output
              << "\" → 清洗后: \"" << cleaned << "\"" << std::endl;

    int action = parseAction(cleaned);
    if (action < 0) {
        action = parseEarlyAction(cleaned);
    }
    if (action < 0) {
        std::cerr << "[TrafficSignScene] VLM 输出无法解析: \""
                  << raw_output << "\" → 清洗后: \"" << cleaned
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

    // ---- 查找 sign class_id（只查一次） ----
    if (traffic_sign_class_id_ < 0) {
        const auto& classes = det_cfg.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "sign") {
                traffic_sign_class_id_ = i;
                std::cout << "[TrafficSignScene] 找到 sign class_id=" << i << std::endl;
                break;
            }
        }
        if (traffic_sign_class_id_ < 0) {
            std::cerr << "[TrafficSignScene] 未找到 sign 类别！classes=[";
            for (size_t i = 0; i < classes.size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << classes[i];
            }
            std::cerr << "]" << std::endl;
        }
    }

    // ---- 获取最新检测结果 ----
    auto det = ctx->det_result.get();

    // ---- 诊断计数器（每60帧打印一次） ----
    static int diag_counter = 0;
    diag_counter++;

    const auto& uc = ctx->config.uart;
    const uint16_t pwm_neutral = static_cast<uint16_t>(uc.pwm_neutral);

    // ---- VLM 三阶段状态机 ----
    switch (vlm_state_) {

    // ================================================================
    // 阶段1：正常巡线 + 检测交通标志，等延时到达后发起 VLM 请求
    // ================================================================
    case VlmState::APPROACH: {
        ctx->vlm_state.store(0, std::memory_order_release); // 通知显示

        const DetectionBox* best = nullptr;
        float best_score = 0.0f;
        int total_boxes = 0;
        if (det && traffic_sign_class_id_ >= 0) {
            total_boxes = static_cast<int>(det->boxes.size());
            for (const auto& b : det->boxes) {
                if (b.class_id == traffic_sign_class_id_ && b.confidence > best_score) {
                    best_score = b.confidence;
                    best = &b;
                }
            }
        }

        if (best && best_score > 0.4f) {
            stable_count_++;
            if (diag_counter % 60 == 0) {
                std::cout << "[TrafficSignScene] sign检测到 best_score=" << best_score
                          << " stable=" << stable_count_ << "/" << vlm_cfg.stable_frames
                          << " box=(" << best->x1 << "," << best->y1
                          << ")-(" << best->x2 << "," << best->y2 << ")"
                          << " total_boxes=" << total_boxes << std::endl;
            }
            if (stable_count_ >= vlm_cfg.stable_frames) {
                delay_counter_++;
                if (delay_counter_ >= vlm_cfg.delay_frames) {
                    cv::Mat work_frame;
                    if (ctx->work_frame_queue.tryConsume(work_frame)) {
                        auto jpeg_bytes = prepareSignImage(
                            work_frame, best->x1, best->y1, best->x2, best->y2, vlm_cfg);

                        // ---- 先停车（串口 + 原子），再发 VLM 请求 ----
                        ctx->uart_io->sendControl(0.0f, pwm_neutral);
                        ctx->target_speed.store(0.0f, std::memory_order_release);
                        ctx->steering_pwm.store(static_cast<float>(pwm_neutral), std::memory_order_release);

                        std::cout << "[TrafficSignScene] 已停车，稳定检测到交通标志 (score="
                                  << best_score << ")，发起 VLM 请求..."
                                  << " 延时" << delay_counter_ << "帧"
                                  << " 图片大小=" << jpeg_bytes.size() / 1024 << "KB"
                                  << std::endl;

                        // ---- 保存 VLM 输入图像（调试用，线程池异步写盘） ----
                        if (ctx->thread_pool) {
                            auto jpeg_copy = jpeg_bytes;
                            ctx->thread_pool->enqueueWithPriority(1, [jpeg = std::move(jpeg_copy)]() {
                                mkdir("./debug_vlm", 0755);
                                auto now = std::chrono::system_clock::now();
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now.time_since_epoch()).count();
                                std::string path = "./debug_vlm/vlm_" + std::to_string(ms) + ".jpg";
                                std::ofstream ofs(path, std::ios::binary);
                                if (ofs) {
                                    ofs.write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size());
                                    std::cout << "[TrafficSignScene] VLM 输入图像已保存: "
                                              << path << std::endl;
                                }
                            });
                        }

                        // 记录请求开始时间，通知显示
                        vlm_request_start_ = std::chrono::steady_clock::now();
                        ctx->vlm_state.store(1, std::memory_order_release); // WAIT_VLM

                        // detached thread + shared_ptr<atomic>：避免 future 析构阻塞
                        vlm_result_ = std::make_shared<std::atomic<int>>(-1);
                        std::thread([jpeg = std::move(jpeg_bytes),
                                     cfg = vlm_cfg,
                                     result = vlm_result_]() {
                            std::promise<int> p;
                            auto f = p.get_future();

                            std::thread([&]() {
                                try {
                                    p.set_value(callVlmApi(jpeg, cfg));
                                } catch (const std::exception& e) {
                                    std::cerr << "[TrafficSignScene] VLM API 异常: "
                                              << e.what() << "，兜底直走" << std::endl;
                                    try { p.set_value(0); } catch (...) {}
                                } catch (...) {
                                    std::cerr << "[TrafficSignScene] VLM API 未知异常，兜底直走"
                                              << std::endl;
                                    try { p.set_value(0); } catch (...) {}
                                }
                            }).detach();

                            auto deadline_ms = std::chrono::milliseconds(
                                static_cast<long>(cfg.deadline_seconds * 1000));

                            if (f.wait_for(deadline_ms) == std::future_status::ready) {
                                result->store(f.get(), std::memory_order_release);
                            } else {
                                std::cerr << "[TrafficSignScene] VLM 请求超时 ("
                                          << cfg.deadline_seconds << "s)，兜底直走" << std::endl;
                                result->store(0, std::memory_order_release);
                            }
                        }).detach();
                        vlm_state_ = VlmState::WAIT_VLM;
                        break;  // 立即跳出 switch，避免下面 approach_follower_.process() 覆盖停车
                    } else {
                        std::cerr << "[TrafficSignScene] 无可用工作帧，跳过 VLM 请求"
                                  << std::endl;
                    }
                    delay_counter_ = 0;
                } else {
                    std::cout << "[TrafficSignScene] 帧延迟 " << delay_counter_
                              << "/" << vlm_cfg.delay_frames << std::endl;
                }
                stable_count_ = 0;
            }
        } else {
            if (stable_count_ > 0 && diag_counter % 60 == 0) {
                std::cout << "[TrafficSignScene] 未检测到sign (best_score="
                          << best_score << " < 0.4 or no sign box)"
                          << " total_boxes=" << total_boxes
                          << " class_id=" << traffic_sign_class_id_ << std::endl;
            }
            stable_count_ = 0;
            delay_counter_ = 0;
        }

        // 阶段1：纯巡线，和 NormalScene 完全一样
        follower_.fork_side = 0;
        ctx->config.control.fork.keep_side = 0;
        follower_.process(seg->mask, ctx);
        break;
    }

    // ================================================================
    // 阶段2：停车等待 VLM 决策返回
    // ================================================================
    case VlmState::WAIT_VLM: {
        // ---- 轮询 VLM 结果 ----
        if (vlm_result_) {
            int r = vlm_result_->load(std::memory_order_acquire);
            if (r >= 0) {
                cached_fork_side_ = r;
                vlm_state_ = VlmState::DRIVE;

                auto now = std::chrono::steady_clock::now();
                auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - vlm_request_start_).count();
                ctx->vlm_latency_ms.store(latency_ms, std::memory_order_release);
                ctx->vlm_decision.store(r, std::memory_order_release);
                ctx->vlm_state.store(2, std::memory_order_release); // DRIVE

                std::cout << "[TrafficSignScene] VLM 决策完成: "
                          << (cached_fork_side_ == 0 ? "直走" : "右转")
                          << " (耗时 " << latency_ms << "ms)" << std::endl;
            }
        }
        // 不调用 follower_.process()，原地停车等待
        break;
    }

    // ================================================================
    // 阶段3：按 VLM 决策执行巡线
    // ================================================================
    case VlmState::DRIVE: {
        follower_.fork_side = cached_fork_side_;
        ctx->config.control.fork.keep_side = cached_fork_side_;

        // 右转：临时扩大 bird_width，让裁剪窗口覆盖整个宽鸟瞰图，
        //       确保右侧岔路在中线提取时可见
        auto saved_bw = ctx->config.calib.bird_width;
        if (cached_fork_side_ == 1) {
            ctx->config.calib.bird_width =
                static_cast<int>(saved_bw * ctx->config.calib.bird_width_scale);
        }
        follower_.process(seg->mask, ctx);
        ctx->config.calib.bird_width = saved_bw;
        break;
    }
    }
}
