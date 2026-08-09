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
void drawSegmentation(cv::Mat& frame, const SegResult& seg,
                      const FrameCropConfig& crop_cfg) {
    if (seg.mask.empty()) return;

    // mask 覆盖区域 = 裁剪区域（全图坐标下）
    int crop_w = crop_cfg.enable ? crop_cfg.w : frame.cols;
    int crop_h = crop_cfg.enable ? crop_cfg.h : frame.rows;
    int crop_x = crop_cfg.enable ? crop_cfg.x : 0;
    int crop_y = crop_cfg.enable ? crop_cfg.y : 0;

    cv::Mat mask_resized;
    cv::resize(seg.mask, mask_resized, cv::Size(crop_w, crop_h), 0, 0, cv::INTER_NEAREST);

    cv::Mat overlay = cv::Mat::zeros(frame.size(), CV_8UC3);
    cv::Mat roi = overlay(cv::Rect(crop_x, crop_y, crop_w, crop_h));
    roi.setTo(kTrackColor, mask_resized);
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

// ---- 绘制鸟瞰图（边界/中线 + 前视点 + 车底中心 + 障碍点 + 中心分界红线） ----
// 注意：调用者需持有 bird_mtx_ 锁
void drawBirdView(cv::Mat& bird_vis, const std::vector<cv::Point2f>& centers,
                  const cv::Point2f& aim_pt, float car_cy_ratio,
                  float car_center_x_offset_ratio,
                  const cv::Point2f* obs_pt) {
    int w = bird_vis.cols, h = bird_vis.rows;

    // 中心分界红线（区分左右两侧，空赛道也画）
    int mid_x = w / 2;
    cv::line(bird_vis, cv::Point(mid_x, 0), cv::Point(mid_x, h - 1),
             cv::Scalar(0, 0, 255), 1);

    // 绘制障碍点（橙色十字，CarScene 检测到 car 时画）
    if (obs_pt) {
        int ox = static_cast<int>(obs_pt->x);
        int oy = static_cast<int>(obs_pt->y);
        cv::line(bird_vis, cv::Point(ox - 6, oy), cv::Point(ox + 6, oy), cv::Scalar(0, 165, 255), 2);
        cv::line(bird_vis, cv::Point(ox, oy - 6), cv::Point(ox, oy + 6), cv::Scalar(0, 165, 255), 2);
        cv::circle(bird_vis, cv::Point(ox, oy), 4, cv::Scalar(0, 165, 255), -1);
    }

    if (centers.empty()) return;

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
    int   vlm_st  = ctx->vlm_state.load(std::memory_order_acquire);
    int   vlm_dec = ctx->vlm_decision.load(std::memory_order_acquire);
    auto  vlm_ms  = ctx->vlm_latency_ms.load(std::memory_order_acquire);
    bool  is_car = (name == "CarScene");
    int   cedge  = ctx->car_edge.load(std::memory_order_acquire);

    // 半透明背景（CarScene 时多一行，加高）
    int hud_h = is_car ? 134 : 112;
    cv::Rect roi(x - 8, 4, 280, hud_h);
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
    y += h;

    // VLM 状态行
    oss.str("");
    if (vlm_st == 0) {
        oss << "VLM: idle";
    } else if (vlm_st == 1) {
        oss << "VLM: requesting...";
    } else if (vlm_st == 2) {
        oss << "VLM: " << (vlm_dec == 0 ? "straight" : "right")
            << " (" << vlm_ms << "ms)";
    }
    cv::putText(frame, oss.str(),
                cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, kHudText, 1);

    // CarScene 沿边方向（仅 CarScene 显示）
    if (is_car) {
        y += h;
        const char* edge_name =
            (cedge > 0) ? "RIGHT edge" : (cedge < 0) ? "LEFT edge" : "CENTER";
        cv::putText(frame, "Car: " + std::string(edge_name),
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, kHudText, 1);
    }
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
                drawSegmentation(display, *seg, ctx_->config.frame_crop);
            }

            auto det = ctx_->det_result.get();
            if (det && !det->boxes.empty()) {
                drawDetections(display, *det, ctx_->config.ai.detection);
            }

            // ---- 2. 右侧：鸟瞰图（赛道mask + 中线 + 前视点） ----
            cv::Mat bird_color;
            {
                std::lock_guard<std::mutex> lock(ctx_->bird_mtx_);
                if (ctx_->bird_valid_ && !ctx_->bird_mask_.empty()) {
                    // 画布尺寸跟随 mask 原生尺寸：CarScene 宽鸟瞰(960) / 其他场景裁切(240)
                    int bird_w = ctx_->bird_mask_.cols;
                    int bird_h = ctx_->bird_mask_.rows;
                    bird_color = cv::Mat(bird_h, bird_w, CV_8UC3, kBirdBg);
                    bird_color.setTo(kBirdTrack, ctx_->bird_mask_);
                    // 车底中心蓝点画到参考点位置：画面中心 - 沿边方向×偏移比例（贴边界内侧）
                    // car_edge 默认 -1（左边界），CarScene 由首帧锁存
                    const auto& sp = ctx_->config.control.scene.at(ctx_->scene_manager.currentName());
                    int cedge = ctx_->car_edge.load(std::memory_order_acquire);
                    if (cedge == 0) cedge = -1;   // 未锁存→默认左边界
                    float off_ratio = -cedge * sp.lateral_offset_ratio;

                    // 障碍点（CarScene 检测到 car 时画）
                    cv::Point2f obs_pt;
                    bool has_obs = false;
                    {
                        std::lock_guard<std::mutex> lk(ctx_->car_obs_mtx_);
                        if (ctx_->car_obs_valid_) { obs_pt = ctx_->car_obs_pt_; has_obs = true; }
                    }
                    drawBirdView(bird_color, ctx_->bird_centers_, ctx_->bird_aim_pt_,
                                 ctx_->config.control.car_cy_ratio, off_ratio,
                                 has_obs ? &obs_pt : nullptr);
                } else {
                    bird_color = cv::Mat(ctx_->config.calib.bird_height,
                                         ctx_->config.calib.bird_width,
                                         CV_8UC3, kBirdBg);
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
