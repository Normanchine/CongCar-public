#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace hal {
namespace memory {

constexpr size_t SHM_HEADER_SIZE = 16;  // fid(uint64) + w(uint32) + h(uint32)

class SharedMemoryManager {
public:
    virtual ~SharedMemoryManager() = default;

    // 打开已存在的共享内存（客户端模式，对接Python服务端）
    virtual bool open(const std::string& name, size_t size) = 0;
    virtual void close() = 0;

    // 返回共享内存首地址（header区域）
    virtual void* getBuffer() const = 0;

    // 返回图像数据起始地址 = getBuffer() + SHM_HEADER_SIZE
    virtual void* getDataBuffer() const = 0;

    virtual bool isValid() const = 0;

    static std::unique_ptr<SharedMemoryManager> create();
};

} // namespace memory
} // namespace hal
