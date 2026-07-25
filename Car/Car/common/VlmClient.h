#pragma once

#include "../config/Config.h"
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

// VLM API 客户端 — 纯网络层，负责图片预处理 → HTTP请求 → SSE流解析 → 响应清洗
// 与场景逻辑完全解耦，可独立测试。

class VlmClient {
public:
    // 从检测框中裁剪交通标志图片，扩边后编码为 JPEG 字节。
    static std::vector<uint8_t> prepareSignImage(const cv::Mat& frame,
                                                   float x1, float y1,
                                                   float x2, float y2,
                                                   const VlmConfig& cfg);

    // 同步调用千帆 VLM API（在后台线程中执行）。
    // @return 0=直走 / 1=右转
    // @throw std::runtime_error 网络故障时抛出；调用方兜底直走。
    static int callVlmApi(const std::vector<uint8_t>& jpeg_bytes,
                          const VlmConfig& cfg);

    // 清理 VLM 输出文本：移除空白和所有中英文标点。
    static std::string cleanOutput(const std::string& text);

private:
    // 解析清洗后的输出：0=直走, 1=右转, -1=无法识别
    static int parseAction(const std::string& cleaned);

    // 前缀匹配，兼容流式截断输出
    static int parseEarlyAction(const std::string& cleaned);
};
