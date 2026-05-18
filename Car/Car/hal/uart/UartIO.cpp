#include "UartIO.h"
#include <cstring>
#include <iostream>
#include <thread>
#include <fcntl.h>      // open()
#include <unistd.h>     // close(), read(), write()
#include <termios.h>    // 串口配置
#include <sys/ioctl.h>  // 串口控制

// ---- 协议常量 ----
static constexpr uint8_t FRAME_HEAD      = 0x42;
static constexpr uint8_t ADDR_CARCTRL    = 0x01;
static constexpr uint8_t ADDR_BUZZER     = 0x04;
static constexpr uint8_t ADDR_KEY        = 0x06;
static constexpr uint8_t CTRL_FRAME_LEN  = 10;   // head+addr+len+speed(4)+pwm(2)+chk

static constexpr uint8_t FRAME_LEN_MIN   = 4;
static constexpr uint8_t FRAME_LEN_MAX   = 12;
static constexpr uint8_t RX_BUF_SIZE     = 64;

// ---- 辅助：校验和 ----
static uint8_t checksum(const uint8_t* data, uint8_t len) {
    uint8_t chk = 0;
    for (uint8_t i = 0; i < len; ++i) chk += data[i];
    return chk;
}

// ============================================================
//  UartIO 实现
// ============================================================

bool UartIO::open(const std::string& port, int baud) {
    close();

    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0) {
        std::cerr << "[UartIO] 打开端口失败: " << port << std::endl;
        return false;
    }

    struct termios opts;
    tcgetattr(fd_, &opts);

    cfsetispeed(&opts, baud);
    cfsetospeed(&opts, baud);

    opts.c_cflag &= ~PARENB;  // 无校验
    opts.c_cflag &= ~CSTOPB;  // 1 停止位
    opts.c_cflag &= ~CSIZE;
    opts.c_cflag |=  CS8;     // 8 数据位
    opts.c_cflag &= ~CRTSCTS; // 无流控
    opts.c_cflag |=  CREAD | CLOCAL;

    opts.c_lflag &= ~(ICANON | ECHO | ISIG);
    opts.c_iflag &= ~(IXON | IXOFF | IXANY);
    opts.c_oflag &= ~OPOST;

    opts.c_cc[VMIN]  = 0;
    opts.c_cc[VTIME] = 5;     // 50ms 超时

    tcsetattr(fd_, TCSANOW, &opts);
    tcflush(fd_, TCIOFLUSH);

    std::cout << "[UartIO] 已打开 " << port << " @" << baud << std::endl;
    return true;
}

void UartIO::close() {
    stopReceive();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void UartIO::sendControl(float speed, uint16_t pwm) {
    if (fd_ < 0) return;

    uint8_t buf[11];
    buf[0] = FRAME_HEAD;
    buf[1] = ADDR_CARCTRL;
    buf[2] = CTRL_FRAME_LEN;

    // speed: float32 little-endian
    union { float f; uint8_t b[4]; } uf;
    uf.f = speed;
    buf[3] = uf.b[0]; buf[4] = uf.b[1];
    buf[5] = uf.b[2]; buf[6] = uf.b[3];

    // pwm: uint16 little-endian
    buf[7] = static_cast<uint8_t>(pwm & 0xFF);
    buf[8] = static_cast<uint8_t>((pwm >> 8) & 0xFF);

    buf[9] = checksum(buf, 9);

    ::write(fd_, buf, 10);
}

void UartIO::sendBuzzer(uint8_t sound) {
    if (fd_ < 0) return;

    uint8_t buf[5];
    buf[0] = FRAME_HEAD;
    buf[1] = ADDR_BUZZER;
    buf[2] = 5;
    buf[3] = sound;
    buf[4] = checksum(buf, 4);

    ::write(fd_, buf, 5);
}

void UartIO::startReceive() {
    if (running_) return;
    running_ = true;
    rx_thread_ = std::thread(&UartIO::receiveLoop, this);
}

void UartIO::stopReceive() {
    running_ = false;
    if (rx_thread_.joinable()) rx_thread_.join();
}

bool UartIO::consumeKeypress() {
    bool v = keypress_.exchange(false, std::memory_order_acquire);
    return v;
}

// ---- 接收线程 ----
void UartIO::receiveLoop() {
    uint8_t buf[RX_BUF_SIZE];

    while (running_) {
        int n = ::read(fd_, buf, 1);
        if (n <= 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        // 同步帧头
        if (buf[0] != FRAME_HEAD) continue;

        // 读地址 + 长度
        uint8_t header[2];
        if (!readBytes(header, 2)) continue;
        uint8_t addr   = header[0];
        uint8_t length = header[1];

        if (length < FRAME_LEN_MIN || length > FRAME_LEN_MAX) continue;

        // 读剩余字节（数据 + 校验和）
        uint8_t remaining = length - 3;  // 已读了 head+addr+len = 3 字节
        uint8_t payload[FRAME_LEN_MAX];
        if (!readBytes(payload, remaining)) continue;

        // 组装完整帧
        uint8_t frame[FRAME_LEN_MAX];
        frame[0] = FRAME_HEAD;
        frame[1] = addr;
        frame[2] = length;
        std::memcpy(frame + 3, payload, remaining);

        // 校验和
        if (checksum(frame, length - 1) != frame[length - 1]) continue;

        // 分发
        if (addr == ADDR_KEY) {
            keypress_.store(true, std::memory_order_release);
        }
    }
}

bool UartIO::readBytes(uint8_t* dst, uint8_t count) {
    uint8_t got = 0;
    int timeout = 20; // 20 × 500μs = 10ms
    while (got < count && timeout > 0) {
        int n = ::read(fd_, dst + got, count - got);
        if (n > 0) { got += n; }
        else { std::this_thread::sleep_for(std::chrono::microseconds(500)); --timeout; }
    }
    return got == count;
}
