#pragma once

#include <memory>
#include <atomic>

// 多读单写双缓冲：写者 publish()，读者 get() 拿 shared_ptr 副本，零阻塞
template <typename T>
class SharedResult {
public:
    void publish(T&& data) {
        std::atomic_store(&data_,
            std::make_shared<const T>(std::move(data)));
    }

    // 返回最新数据的只读副本，无数据时返回 nullptr
    std::shared_ptr<const T> get() const {
        return std::atomic_load(&data_);
    }

private:
    std::shared_ptr<const T> data_;
};
