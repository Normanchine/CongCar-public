#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

class UartIO {
public:
    UartIO() = default;
    ~UartIO() { close(); }

    UartIO(const UartIO&) = delete;
    UartIO& operator=(const UartIO&) = delete;

    bool open(const std::string& port, int baud);
    void close();

    void sendControl(float speed, uint16_t pwm);
    void sendBuzzer(uint8_t sound);

    void startReceive();
    void stopReceive();

    bool consumeKeypress();

    // ---- IMU 数据（下位机通过 addr=0x09 发送） ----
    float getYaw() const { return yaw_.load(std::memory_order_acquire); }
    float getRoll()  const { return roll_.load(std::memory_order_acquire); }
    float getPitch() const { return pitch_.load(std::memory_order_acquire); }

private:
    void receiveLoop();
    bool readBytes(uint8_t* dst, uint8_t count);

    int  fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread rx_thread_;
    std::atomic<bool> keypress_{false};
    std::atomic<float> yaw_{0.0f};     // 航向角 (度)
    std::atomic<float> roll_{0.0f};    // 横滚角 (度)
    std::atomic<float> pitch_{0.0f};   // 俯仰角 (度)
};
