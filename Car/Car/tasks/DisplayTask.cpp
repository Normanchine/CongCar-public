#include "DisplayTask.h"
#include "../common/PerfStats.h"
#include "../common/SharedResult.h"
#include "../thread/ThreadTaskFactory.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <iomanip>

REGISTER_THREAD_TASK(DisplayTask)

DisplayTask::DisplayTask(SharedContext* ctx) : IThreadTask(ctx) {}
DisplayTask::~DisplayTask() = default;

namespace {

constexpr float kSegAlpha = 0.3f;
const cv::Scalar kTrackColor(0, 255, 0);
const cv::Scalar kBoxColor(255, 0, 255);
const cv::Scalar kLabelColor(0, 230, 230);

// HUD
const cv::Scalar kHudBg(30, 30, 30);
const cv::Scalar kHudText(220, 220, 220);
constexpr int kPipW = 160, kPipH = 120;

void drawSegmentation(cv::Mat& frame, const SegResult& seg) {
    cv::Mat mask = seg.mask;
    if (mask.size() != frame.size()) {
        cv::resize(mask, mask, frame.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat overlay = cv::Mat::zeros(frame.size(), CV_8UC3);
    overlay.setTo(kTrackColor, mask);
    cv::addWeighted(frame, 1.0 - kSegAlpha, overlay, kSegAlpha, 0, frame);
}

void drawDetections(cv::Mat& frame, const DetResult& det) {
    for (const auto& box : det.boxes) {
        auto p1 = cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1));
        auto p2 = cv::Point(static_cast<int>(box.x2), static_cast<int>(box.y2));
        cv::rectangle(frame, p1, p2, kBoxColor, 2);
        std::ostringstream oss;
        oss << box.class_id << " " << std::fixed << std::setprecision(2) << box.confidence;
        cv::putText(frame, oss.str(),
                    cv::Point(p1.x, p1.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kLabelColor, 2);
    }
}

void drawHUD(cv::Mat& frame, SharedContext* ctx) {
    const int x = 12, h = 22;
    int y = 30;

    const auto& name = ctx->scene_manager.currentName();
    float pwm = ctx->steering_pwm.load(std::memory_order_acquire);
    float spd = ctx->target_speed.load(std::memory_order_acquire);

    // 半透明背景
    cv::Rect roi(x - 8, 8, 240, 100);
    cv::Mat bg = frame(roi);
    cv::Mat dark(roi.height, roi.width, CV_8UC3, kHudBg);
    cv::addWeighted(dark, 0.55, bg, 0.45, 0, bg);

    cv::putText(frame, "Scene: " + name,
                cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, kHudText, 1);
    y += h;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << "PWM: " << pwm
        << "  Speed: " << std::setprecision(2) << spd << " m/s";
    cv::putText(frame, oss.str(),
                cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, kHudText, 1);
    y += h;
    double seg_fps = ctx->perf_stats.getFps(SLOT_SEGMENTATION);
    double det_fps = ctx->perf_stats.getFps(SLOT_DETECTION);
    oss.str("");
    oss << std::setprecision(1) << "Seg: " << seg_fps
        << " fps  Det: " << det_fps << " fps";
    cv::putText(frame, oss.str(),
                cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, kHudText, 1);
}

void drawPIP(cv::Mat& frame, SharedContext* ctx) {
    std::lock_guard<std::mutex> lock(ctx->bird_mtx_);
    if (!ctx->bird_valid_ || ctx->bird_mask_.empty()) return;

    // 缩放鸟瞰 mask 到 PIP 尺寸
    cv::Mat pip;
    cv::resize(ctx->bird_mask_, pip, cv::Size(kPipW, kPipH), 0, 0, cv::INTER_NEAREST);

    cv::Mat pip_color(kPipH, kPipW, CV_8UC3);
    pip_color.setTo(cv::Scalar(30, 30, 30));           // 背景色
    pip_color.setTo(cv::Scalar(0, 200, 0), pip);       // 赛道=绿色

    // 缩放中线坐标
    float sx = static_cast<float>(kPipW) / ctx->bird_mask_.cols;
    float sy = static_cast<float>(kPipH) / ctx->bird_mask_.rows;
    for (const auto& pt : ctx->bird_centers_) {
        cv::Point ip(static_cast<int>(pt.x * sx), static_cast<int>(pt.y * sy));
        cv::circle(pip_color, ip, 1, cv::Scalar(0, 255, 255), -1);  // 黄色中线点
    }
    // 前视点
    cv::Point ap(static_cast<int>(ctx->bird_aim_pt_.x * sx),
                 static_cast<int>(ctx->bird_aim_pt_.y * sy));
    cv::circle(pip_color, ap, 3, cv::Scalar(0, 0, 255), -1);        // 红色前视点

    // 贴到右下角
    int px = frame.cols - kPipW - 12;
    int py = frame.rows - kPipH - 12;
    cv::Rect roi(px, py, kPipW, kPipH);
    cv::addWeighted(pip_color, 0.8, frame(roi), 0.2, 0, frame(roi));
}

} // namespace

void DisplayTask::run() {
    // 启动HTTP推流服务器
    if (ctx_->config.http_stream.enable) {
        http_server_ = std::make_unique<HttpStreamServer>(
            ctx_->config.http_stream.port,
            ctx_->config.http_stream.jpeg_quality
        );
        
        if (http_server_->start()) {
            std::cout << "[DisplayTask] HTTP推流服务器启动，端口: " 
                      << ctx_->config.http_stream.port << std::endl;
            std::cout << "[DisplayTask] 浏览器访问: http://<开发板IP>:" 
                      << ctx_->config.http_stream.port << std::endl;
        } else {
            std::cerr << "[DisplayTask] HTTP推流服务器启动失败" << std::endl;
            http_server_.reset();
        }
    }

    while (!ctx_->stop.load(std::memory_order_acquire)) {

        cv::Mat frame;
        auto t_start = std::chrono::steady_clock::now();

        if (ctx_->display_queue.tryConsume(frame)) {

            // 1. 语义分割覆盖
            auto seg = ctx_->seg_result.get();
            if (seg && !seg->mask.empty()) {
                drawSegmentation(frame, *seg);
            }

            // 2. 目标检测框
            auto det = ctx_->det_result.get();
            if (det && !det->boxes.empty()) {
                drawDetections(frame, *det);
            }

            // 3. HUD + PIP
            drawHUD(frame, ctx_);
            drawPIP(frame, ctx_);

            // 4. HTTP推流（替换imshow）
            if (http_server_ && http_server_->isRunning()) {
                http_server_->pushFrame(frame);
            }

            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_end - t_start).count();
            ctx_->perf_stats.record(SLOT_DISPLAY, elapsed_us);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    // 停止HTTP服务器
    if (http_server_) {
        http_server_->stop();
        http_server_.reset();
    }

    std::cout << "[DisplayTask] 已退出" << std::endl;
}
