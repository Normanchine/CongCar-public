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

private:
    void receiveLoop();
    bool readBytes(uint8_t* dst, uint8_t count);

    int  fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread rx_thread_;
    std::atomic<bool> keypress_{false};
};
