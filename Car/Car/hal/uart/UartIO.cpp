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
static constexpr uint8_t ADDR_IMU        = 0x09;   // IMU姿态数据（下位机→上位机）
static constexpr uint8_t CTRL_FRAME_LEN  = 10;   // head+addr+len+speed(4)+pwm(2)+chk

static constexpr uint8_t FRAME_LEN_MIN   = 4;
static constexpr uint8_t FRAME_LEN_MAX   = 12;
static constexpr uint8_t RX_BUF_SIZE     = 64;

// ---- 波特率映射 ----
static speed_t baudToSpeed(int baud) {
    switch (baud) {
        case 9600:    return B9600;
        case 19200:   return B19200;
        case 38400:   return B38400;
        case 57600:   return B57600;
        case 115200:  return B115200;
        case 230400:  return B230400;
        case 460800:  return B460800;
        case 921600:  return B921600;
        default:      return B115200;
    }
}

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
    if (tcgetattr(fd_, &opts) < 0) {
        std::cerr << "[UartIO] tcgetattr 失败" << std::endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    speed_t speed = baudToSpeed(baud);
    if (cfsetispeed(&opts, speed) < 0 || cfsetospeed(&opts, speed) < 0) {
        std::cerr << "[UartIO] 波特率设置失败" << std::endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

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
    opts.c_cc[VTIME] = 5;     // 0.5s 超时

    if (tcsetattr(fd_, TCSANOW, &opts) < 0) {
        std::cerr << "[UartIO] tcsetattr 失败" << std::endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }
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

    uint8_t buf[10];
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

    ssize_t written = ::write(fd_, buf, 10);
    if (written != 10) {
        std::cerr << "[UartIO] sendControl write 失败: " << written << std::endl;
    }
}

void UartIO::sendBuzzer(uint8_t sound) {
    if (fd_ < 0) return;

    uint8_t buf[5];
    buf[0] = FRAME_HEAD;
    buf[1] = ADDR_BUZZER;
    buf[2] = 5;
    buf[3] = sound;
    buf[4] = checksum(buf, 4);

    ssize_t written = ::write(fd_, buf, 5);
    if (written != 5) {
        std::cerr << "[UartIO] sendBuzzer write 失败: " << written << std::endl;
    }
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
        } else if (addr == ADDR_IMU && length == 10) {
            // IMU帧: head+addr+len + 6字节数据(3×int16: roll/pitch/yaw) + chk
            // 维特智能IMU I2C读取为小端序 [低字节, 高字节]，需 (高<<8)|低
            // 官方SDK: sReg[i] = (buff[i*2+1]<<8) | buff[i*2]
            auto parseI16LE = [](const uint8_t* p) -> int16_t {
                return static_cast<int16_t>((static_cast<uint16_t>(p[1]) << 8) | p[0]);
            };
            int16_t roll_raw  = parseI16LE(&payload[0]);
            int16_t pitch_raw = parseI16LE(&payload[2]);
            int16_t yaw_raw   = parseI16LE(&payload[4]);

            constexpr float kAngleScale = 180.0f / 32768.0f;
            float r = static_cast<float>(roll_raw) * kAngleScale;
            float p = static_cast<float>(pitch_raw) * kAngleScale;
            float y = static_cast<float>(yaw_raw) * kAngleScale;
            roll_.store(r, std::memory_order_release);
            pitch_.store(p, std::memory_order_release);
            yaw_.store(y, std::memory_order_release);

            // 调试打印：前10帧 + 之后每100帧打印一次
            static int imu_cnt = 0;
            ++imu_cnt;
            if (imu_cnt <= 10 || imu_cnt % 100 == 0) {
                std::cout << "[IMU] #" << imu_cnt
                          << " roll=" << r << "° pitch=" << p << "° yaw=" << y
                          << "° (raw: " << roll_raw << "," << pitch_raw << "," << yaw_raw << ")"
                          << std::endl;
            }
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
