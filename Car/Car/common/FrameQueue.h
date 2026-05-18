#pragma once

#include <mutex>
#include <deque>
#include <thread>
#include <chrono>

template <typename T>
class FrameQueue {
private:
    std::deque<T> buffer_;
    int buffer_size_;
    std::mutex mtx_;

public:
    explicit FrameQueue(int size) : buffer_size_(size) {}

    FrameQueue(const FrameQueue&) = delete;
    FrameQueue& operator=(const FrameQueue&) = delete;

    ~FrameQueue() = default;

    bool consume(T& product);
    bool tryConsume(T& product);
    bool produce(const T& product);
};

template <typename T>
bool FrameQueue<T>::consume(T& product) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx_);

        if (!buffer_.empty()) {
            product = std::move(buffer_.front());
            buffer_.pop_front();
            return true;
        }

        lock.unlock();
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

template <typename T>
bool FrameQueue<T>::tryConsume(T& product) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!buffer_.empty()) {
        product = std::move(buffer_.front());
        buffer_.pop_front();
        return true;
    }
    return false;
}

template <typename T>
bool FrameQueue<T>::produce(const T& product) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (buffer_.size() < static_cast<size_t>(buffer_size_)) {
        buffer_.push_back(product);
    } else {
        buffer_.pop_front();
        buffer_.push_back(product);
    }

    return true;
}
