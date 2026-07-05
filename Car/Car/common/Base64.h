#pragma once

#include <cstdint>
#include <string>
#include <vector>

// header-only Base64 编码器（RFC 4648）
namespace base64 {

static const char kTable[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

inline std::string encode(const std::vector<uint8_t>& data) {
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    while (i + 2 < data.size()) {
        uint32_t triple = (static_cast<uint32_t>(data[i]) << 16)
                        | (static_cast<uint32_t>(data[i + 1]) << 8)
                        | static_cast<uint32_t>(data[i + 2]);
        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        out.push_back(kTable[(triple >> 6) & 0x3F]);
        out.push_back(kTable[triple & 0x3F]);
        i += 3;
    }

    if (i < data.size()) {
        uint32_t triple = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < data.size())
            triple |= static_cast<uint32_t>(data[i + 1]) << 8;

        out.push_back(kTable[(triple >> 18) & 0x3F]);
        out.push_back(kTable[(triple >> 12) & 0x3F]);
        if (i + 1 < data.size())
            out.push_back(kTable[(triple >> 6) & 0x3F]);
        else
            out.push_back('=');
        out.push_back('=');
    }

    return out;
}

} // namespace base64
