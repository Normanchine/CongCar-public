#include "shared_memory_manager.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

namespace hal {
namespace memory {

class SharedMemoryImpl : public SharedMemoryManager {
public:
    ~SharedMemoryImpl() override { close(); }

    bool open(const std::string& name, size_t size) override {
        fd_ = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd_ == -1) return false;

        addr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        name_ = name;
        size_ = size;
        return true;
    }

    void close() override {
        if (addr_ && addr_ != MAP_FAILED) {
            munmap(addr_, size_);
            addr_ = nullptr;
        }
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void* getBuffer() const override {
        return addr_;
    }

    void* getDataBuffer() const override {
        return static_cast<uint8_t*>(addr_) + SHM_HEADER_SIZE;
    }

    bool isValid() const override {
        return addr_ != nullptr && addr_ != MAP_FAILED;
    }

private:
    std::string name_;
    size_t size_ = 0;
    int fd_ = -1;
    void* addr_ = nullptr;
};

std::unique_ptr<SharedMemoryManager> SharedMemoryManager::create() {
    return std::make_unique<SharedMemoryImpl>();
}

} // namespace memory
} // namespace hal
