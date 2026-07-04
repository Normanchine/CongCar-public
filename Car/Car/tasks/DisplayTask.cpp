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
const cv::Scalar kCenterColor(0, 255, 255);   // 黄色中线点
const cv::Scalar kAimColor(0, 0, 255);        // 红色前视点
const cv::Scalar kCarColor(255, 0, 0);        // 蓝色车底中心
const cv::Scalar kBirdBg(30, 30, 30);         // 鸟瞰背景色
const cv::Scalar kBirdTrack(0, 200, 0);       // 鸟瞰赛道色

// HUD
const cv::Scalar kHudBg(30, 30, 30);
const cv::Scalar kHudText(220, 220, 220);

// ---- 绘制分割叠加 ----
void drawSegmentation(cv::Mat& frame, const SegResult& seg) {
    cv::Mat mask = seg.mask;
    if (mask.size() != frame.size()) {
        cv::resize(mask, mask, frame.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat overlay = cv::Mat::zeros(frame.size(), CV_8UC3);
    overlay.setTo(kTrackColor, mask);
    cv::addWeighted(frame, 1.0 - kSegAlpha, overlay, kSegAlpha, 0, frame);
}

// ---- 绘制检测框 ----
void drawDetections(cv::Mat& frame, const DetResult& det, const DetectionConfig& det_cfg) {
    int num = det_cfg.class_num;
    if (num <= 0) return;
    const auto& names  = det_cfg.classes;
    const auto& colors = det_cfg.colors;

    for (const auto& box : det.boxes) {
        int x1 = static_cast<int>(box.x1);
        int y1 = static_cast<int>(box.y1);
        int x2 = static_cast<int>(box.x2);
        int y2 = static_cast<int>(box.y2);
        int cls = box.class_id;
        if (cls < 0 || cls >= num) continue;

        cv::Scalar color(colors[cls][0], colors[cls][1], colors[cls][2]);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", names[cls].c_str(), box.confidence);
        int baseline = 0;
        cv::Size text_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5f, 1, &baseline);
        cv::rectangle(frame, cv::Point(x1, y1 - text_sz.height - 4),
                      cv::Point(x1 + text_sz.width + 4, y1), color, -1);
        cv::putText(frame, label, cv::Point(x1 + 2, y1 - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(255, 255, 255), 1);
    }
}

// ---- 绘制鸟瞰图（中线 + 前视点 + 车底中心） ----
// 注意：调用者需持有 bird_mtx_ 锁
void drawBirdView(cv::Mat& bird_vis, const std::vector<cv::Point2f>& centers,
                  const cv::Point2f& aim_pt, float car_cy_ratio,
                  float car_center_x_offset_ratio) {
    if (centers.empty()) return;

    int w = bird_vis.cols, h = bird_vis.rows;
    float cx = w * (0.5f + car_center_x_offset_ratio);
    float cy = h * car_cy_ratio;

    // 绘制中线（黄色小点）
    for (const auto& pt : centers) {
        cv::circle(bird_vis, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                   2, kCenterColor, -1);
    }
    // 绘制前视点（红色大圈）
    cv::circle(bird_vis, cv::Point(static_cast<int>(aim_pt.x), static_cast<int>(aim_pt.y)),
               5, kAimColor, -1);
    // 绘制车底中心（蓝色圈）
    cv::circle(bird_vis, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
               6, kCarColor, 2);
}

// ---- 绘制HUD ----
void drawHUD(cv::Mat& frame, SharedContext* ctx) {
    const int x = 12, h = 22;
    int y = 24;

    const auto& name = ctx->scene_manager.currentName();
    float pwm = ctx->steering_pwm.load(std::memory_order_acquire);
    float spd = ctx->target_speed.load(std::memory_order_acquire);

    // 半透明背景
    cv::Rect roi(x - 8, 4, 260, 88);
    if (roi.x + roi.width <= frame.cols && roi.y + roi.height <= frame.rows) {
        cv::Mat bg = frame(roi);
        cv::Mat dark(roi.height, roi.width, CV_8UC3, kHudBg);
        cv::addWeighted(dark, 0.55, bg, 0.45, 0, bg);
    }

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

            // ---- 1. 左侧：原图 + 分割叠加 + 检测框 ----
            cv::Mat display = frame.clone();

            auto seg = ctx_->seg_result.get();
            if (seg && !seg->mask.empty()) {
                drawSegmentation(display, *seg);
            }

            auto det = ctx_->det_result.get();
            if (det && !det->boxes.empty()) {
                drawDetections(display, *det, ctx_->config.ai.detection);
            }

            // ---- 2. 右侧：鸟瞰图（赛道mask + 中线 + 前视点） ----
            int bird_w = ctx_->config.calib.bird_width;
            int bird_h = ctx_->config.calib.bird_height;
            cv::Mat bird_color(bird_h, bird_w, CV_8UC3);

            {
                std::lock_guard<std::mutex> lock(ctx_->bird_mtx_);
                if (ctx_->bird_valid_ && !ctx_->bird_mask_.empty()) {
                    bird_color.setTo(kBirdBg);
                    bird_color.setTo(kBirdTrack, ctx_->bird_mask_);
                    float center_offset_ratio = 0.0f;
                    const auto& scene_name = ctx_->scene_manager.currentName();
                    auto it = ctx_->config.control.scene.find(scene_name);
                    if (scene_name == "CarScene" && it != ctx_->config.control.scene.end()) {
                        center_offset_ratio = std::clamp(
                            it->second.car_center_x_offset_ratio, -0.45f, 0.45f);
                    }
                    drawBirdView(bird_color, ctx_->bird_centers_, ctx_->bird_aim_pt_,
                                 ctx_->config.control.car_cy_ratio, center_offset_ratio);
                } else {
                    bird_color.setTo(kBirdBg);
                }
            }
            // BGR→RGB，与左侧display统一色域后再拼接
            cv::cvtColor(bird_color, bird_color, cv::COLOR_BGR2RGB);

            // ---- 3. 水平拼接：左(原图+分割+检测) | 右(鸟瞰图) ----
            int target_h = display.rows;
            if (bird_color.rows != target_h) {
                cv::resize(bird_color, bird_color, cv::Size(
                    static_cast<int>(static_cast<float>(bird_color.cols) * target_h / bird_color.rows),
                    target_h));
            }
            cv::Mat combined;
            cv::hconcat(display, bird_color, combined);

            // ---- 3.5 岔路mask对比：处理前 | 处理后 ----
            {
                std::lock_guard<std::mutex> lock(ctx_->fork_mtx_);
                if (ctx_->fork_valid_) {
                    int mw = 240, mh = static_cast<int>(static_cast<float>(mw)
                        * ctx_->fork_raw_mask_.rows / ctx_->fork_raw_mask_.cols);
                    auto renderMask = [&](const cv::Mat& m) -> cv::Mat {
                        cv::Mat resized, vis;
                        cv::resize(m, resized, cv::Size(mw, mh), 0, 0, cv::INTER_NEAREST);
                        vis = cv::Mat(mh, mw, CV_8UC3);
                        vis.setTo(kBirdBg);
                        vis.setTo(kBirdTrack, resized);
                        return vis;
                    };

                    cv::Mat raw_vis = renderMask(ctx_->fork_raw_mask_);
                    cv::putText(raw_vis, "raw", cv::Point(4, 14),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, kHudText, 1);

                    cv::Mat done_vis = renderMask(ctx_->fork_done_mask_);
                    cv::putText(done_vis, "done", cv::Point(4, 14),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, kHudText, 1);

                    cv::Mat fork_row;
                    cv::hconcat(raw_vis, done_vis, fork_row);
                    // 调整宽度与combined一致
                    if (fork_row.cols != combined.cols) {
                        cv::resize(fork_row, fork_row,
                                   cv::Size(combined.cols, fork_row.rows));
                    }
                    cv::vconcat(combined, fork_row, combined);
                }
            }

            // ---- 4. HUD叠加 ----
            drawHUD(combined, ctx_);

            // ---- 5. HTTP推流（RGB→BGR，OpenCV JPEG编码需要BGR） ----
            if (http_server_ && http_server_->isRunning()) {
                cv::Mat combined_bgr;
                cv::cvtColor(combined, combined_bgr, cv::COLOR_RGB2BGR);
                http_server_->pushFrame(combined_bgr);
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
