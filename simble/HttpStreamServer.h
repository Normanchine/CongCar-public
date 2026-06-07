//
// HttpStreamServer.h - HTTP MJPEG推流服务器（纯socket实现）
// 功能：在独立线程中运行HTTP服务器，推送MJPEG视频流到浏览器
// 特点：无外部依赖，性能优异，适合嵌入式环境
// 作者：嵌入式开发工程师
// 日期：2024
//

#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <thread>
#include <iostream>

// ============================================================================
// 帧缓冲队列（线程安全）
// ============================================================================
class FrameBuffer {
public:
    // 构造函数
    // @param max_size 队列最大容量
    explicit FrameBuffer(size_t max_size = 3)
        : max_size_(max_size), stopped_(false) {}

    // 析构函数
    ~FrameBuffer() = default;

    // 推入帧（非阻塞，队列满时丢弃最旧帧）
    // @param frame 图像帧（BGR格式）
    // @return 成功返回true
    bool push(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (stopped_) return false;

        // 队列满时移除最旧帧
        if (queue_.size() >= max_size_) {
            queue_.pop();
        }

        queue_.push(frame.clone());
        cond_.notify_one();
        return true;
    }

    // 取出帧（阻塞等待）
    // @param frame 输出图像帧
    // @return 成功返回true，停止时返回false
    bool pop(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mutex_);

        cond_.wait(lock, [this] { return !queue_.empty() || stopped_; });

        if (stopped_) return false;

        frame = queue_.front();
        queue_.pop();
        return true;
    }

    // 停止队列
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cond_.notify_all();
    }

    // 重置队列
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
        stopped_ = false;
    }

private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    size_t max_size_;
    bool stopped_;
};

// ============================================================================
// HTTP MJPEG推流服务器
// ============================================================================
class HttpStreamServer {
public:
    // 构造函数
    // @param port HTTP服务器端口
    // @param jpeg_quality JPEG编码质量（1-100）
    HttpStreamServer(int port = 5001, int jpeg_quality = 80)
        : port_(port)
        , jpeg_quality_(jpeg_quality)
        , server_fd_(-1)
        , running_(false)
        , frame_buffer_(3) {}

    // 析构函数
    ~HttpStreamServer() {
        stop();
    }

    // 启动服务器
    // @return 成功返回true
    bool start() {
        if (running_.load()) return true;

        // 创建socket
        server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            perror("[HTTP] socket创建失败");
            return false;
        }

        // 设置端口复用
        int opt = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        // 绑定地址
        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(server_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("[HTTP] bind失败");
            ::close(server_fd_);
            server_fd_ = -1;
            return false;
        }

        // 监听
        if (listen(server_fd_, 5) < 0) {
            perror("[HTTP] listen失败");
            ::close(server_fd_);
            server_fd_ = -1;
            return false;
        }

        running_.store(true);

        // 启动接受连接线程
        pthread_t tid;
        pthread_create(&tid, nullptr, acceptThread, this);
        pthread_detach(tid);

        std::cout << "[HTTP] 服务器启动成功，端口: " << port_ << std::endl;
        return true;
    }

    // 停止服务器
    void stop() {
        if (!running_.load()) return;

        running_.store(false);
        frame_buffer_.stop();

        if (server_fd_ >= 0) {
            ::close(server_fd_);
            server_fd_ = -1;
        }

        std::cout << "[HTTP] 服务器已停止" << std::endl;
    }

    // 推送帧到缓冲队列
    // @param frame 图像帧（BGR格式）
    void pushFrame(const cv::Mat& frame) {
        frame_buffer_.push(frame);
    }

    // 检查服务器是否运行中
    bool isRunning() const {
        return running_.load();
    }

    // 获取服务器端口
    int getPort() const {
        return port_;
    }

private:
    int port_;                      // HTTP端口
    int jpeg_quality_;              // JPEG编码质量
    int server_fd_;                 // 服务器socket
    std::atomic<bool> running_;     // 运行标志
    FrameBuffer frame_buffer_;      // 帧缓冲队列

    // 接受客户端连接线程（静态函数）
    static void* acceptThread(void* arg) {
        HttpStreamServer* self = static_cast<HttpStreamServer*>(arg);
        self->acceptLoop();
        return nullptr;
    }

    // 接受连接循环
    void acceptLoop() {
        while (running_.load()) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);

            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (running_.load()) {
                    perror("[HTTP] accept失败");
                }
                continue;
            }

            // 获取客户端IP
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
            std::cout << "[HTTP] 新客户端连接: " << client_ip << std::endl;

            // 创建客户端处理线程
            pthread_t tid;
            pthread_create(&tid, nullptr, clientThread, new ClientContext{this, client_fd});
            pthread_detach(tid);
        }
    }

    // 客户端上下文
    struct ClientContext {
        HttpStreamServer* server;
        int client_fd;
    };

    // 客户端处理线程（静态函数）
    static void* clientThread(void* arg) {
        ClientContext* ctx = static_cast<ClientContext*>(arg);
        HttpStreamServer* server = ctx->server;
        int client_fd = ctx->client_fd;
        delete ctx;

        server->handleClient(client_fd);
        return nullptr;
    }

    // 处理客户端请求
    void handleClient(int client_fd) {
        // 读取HTTP请求（简单解析）
        char buffer[1024] = {0};
        ssize_t n = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (n <= 0) {
            ::close(client_fd);
            return;
        }

        // 解析请求路径
        std::string request(buffer);
        std::string path = "/video_feed";  // 默认路径

        if (request.find("GET / ") != std::string::npos) {
            path = "/";
        } else if (request.find("GET /video_feed") != std::string::npos) {
            path = "/video_feed";
        }

        // 根据路径返回响应
        if (path == "/") {
            sendIndexPage(client_fd);
        } else if (path == "/video_feed") {
            sendVideoStream(client_fd);
        } else {
            send404(client_fd);
        }

        ::close(client_fd);
    }

    // 发送主页HTML
    void sendIndexPage(int client_fd) {
        std::string html = R"(
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simble远程监控</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        img {
            width: 100vw;
            height: auto;
            object-fit: contain;
        }
    </style>
</head>
<body>
    <img src="/video_feed" alt="Simble监控">
</body>
</html>
)";

        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: " + std::to_string(html.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n" + html;

        send(client_fd, response.c_str(), response.size(), 0);
    }

    // 发送MJPEG视频流
    void sendVideoStream(int client_fd) {
        // 发送HTTP头部
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "\r\n";

        if (send(client_fd, header.c_str(), header.size(), 0) < 0) {
            std::cout << "[HTTP] 发送头部失败" << std::endl;
            return;
        }

        std::cout << "[HTTP] 开始推流" << std::endl;

        // 重置帧缓冲
        frame_buffer_.reset();

        // 持续发送帧
        cv::Mat frame;
        while (running_.load() && frame_buffer_.pop(frame)) {
            // JPEG编码
            std::vector<uchar> buffer;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};

            if (!cv::imencode(".jpg", frame, buffer, params)) {
                std::cerr << "[HTTP] JPEG编码失败" << std::endl;
                continue;
            }

            // 构造MJPEG帧
            std::string boundary = "--frame\r\n";
            std::string content_type = "Content-Type: image/jpeg\r\n";
            std::string content_length = "Content-Length: " + std::to_string(buffer.size()) + "\r\n\r\n";

            // 发送boundary
            if (send(client_fd, boundary.c_str(), boundary.size(), MSG_NOSIGNAL) < 0) {
                break;
            }

            // 发送Content-Type
            if (send(client_fd, content_type.c_str(), content_type.size(), MSG_NOSIGNAL) < 0) {
                break;
            }

            // 发送Content-Length
            if (send(client_fd, content_length.c_str(), content_length.size(), MSG_NOSIGNAL) < 0) {
                break;
            }

            // 发送JPEG数据
            if (send(client_fd, buffer.data(), buffer.size(), MSG_NOSIGNAL) < 0) {
                break;
            }

            // 发送结束标记
            if (send(client_fd, "\r\n", 2, MSG_NOSIGNAL) < 0) {
                break;
            }

            // 短暂休眠，避免CPU占用过高
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::cout << "[HTTP] 推流结束" << std::endl;
    }

    // 发送404响应
    void send404(int client_fd) {
        std::string html = "<h1>404 Not Found</h1>";

        std::string response =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: " + std::to_string(html.size()) + "\r\n"
            "Connection: close\r\n"
            "\r\n" + html;

        send(client_fd, response.c_str(), response.size(), 0);
    }
};
