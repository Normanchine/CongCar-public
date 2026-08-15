#include "VlmClient.h"
#include "VlmPrompt.h"
#include "Base64.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

// ---- libcurl 写回调 ----
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

// ---- 清理 VLM 输出文本 ----
std::string VlmClient::cleanOutput(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char ch : text) {
        if (ch == '\r' || ch == '\n' || ch == '\t' || ch == ' ') continue;
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
int VlmClient::parseAction(const std::string& cleaned) {
    if (cleaned == "直走") return 0;
    if (cleaned == "右转") return 1;
    return -1;
}

// ---- 找最后出现的动作词（思考+动作分隔文本 / 流式截断都兼容） ----
int VlmClient::findLastAction(const std::string& cleaned) {
    size_t zc = cleaned.rfind("直走");
    size_t zz = cleaned.rfind("右转");
    if (zc == std::string::npos && zz == std::string::npos) return -1;
    if (zc == std::string::npos) return 1;
    if (zz == std::string::npos) return 0;
    return (zc > zz) ? 0 : 1;
}

// ---- 准备发送给 VLM 的图片 ----
std::vector<uint8_t> VlmClient::prepareSignImage(const cv::Mat& frame,
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

// ---- 同步调用千帆 VLM API ----
int VlmClient::callVlmApi(const std::vector<uint8_t>& jpeg_bytes,
                            const VlmConfig& cfg) {
    std::string api_key = resolveApiKey(cfg);
    if (api_key.empty())
        throw std::runtime_error("未配置 VLM API Key（config.vlm.api_key）");

    std::cout << "[VlmClient] 请求: url=" << (cfg.base_url + "/chat/completions")
              << " model=" << cfg.model
              << " deadline=" << cfg.deadline_seconds << "s"
              << " key_prefix=" << api_key.substr(0, 12) << "..." << std::endl;

    // 1. Base64 编码 → data URL
    std::string b64 = base64::encode(jpeg_bytes);
    std::string image_url = "data:image/jpeg;base64," + b64;

    // 2. 构造请求体（stream=true）
    json user_content = json::array({
        {
            {"type", "image_url"},
            {"image_url", {
                {"url", image_url},
                {"detail", "high"},
            }},
        },
        {
            {"type", "text"},
            {"text", "请先报出你识别到的牌面文字，再简要说明推理过程，最后给出最终动作。注意先读完整句：近道、捷径、更短路线类表达通常指右转，但被否定或被\"才怪\"等推翻时不能按右转处理。"},
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
        {"max_tokens",  cfg.max_tokens},
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

    // 4. 解析 SSE 流式响应
    std::string raw_output;
    {
        std::istringstream stream(response_buf.data);
        std::string line;
        while (std::getline(stream, line)) {
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

    std::cout << "[VlmClient] 原始输出: \"" << raw_output
              << "\" → 清洗后: \"" << cleaned << "\"" << std::endl;

    int action = parseAction(cleaned);
    if (action < 0) {
        action = findLastAction(cleaned);
    }
    if (action < 0) {
        std::cerr << "[VlmClient] 输出无法解析: \""
                  << raw_output << "\" → 清洗后: \"" << cleaned
                  << "\"，兜底直走" << std::endl;
        return 0;
    }

    return action;
}
