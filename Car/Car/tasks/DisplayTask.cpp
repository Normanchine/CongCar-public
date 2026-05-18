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
#include <iomanip>  // 加上这一行
REGISTER_THREAD_TASK(DisplayTask)

DisplayTask::DisplayTask(SharedContext* ctx) : IThreadTask(ctx) {}
DisplayTask::~DisplayTask() = default;

namespace {

constexpr float kSegAlpha = 0.3f;

const cv::Scalar kTrackColor(0, 255, 0);    // 绿色赛道覆盖
const cv::Scalar kBoxColor(255, 0, 255);    // 紫色检测框
const cv::Scalar kLabelColor(0, 230, 230);  // 黄色标签

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

} // anon namespace

void DisplayTask::run() {
    const char* kWindowName = "Car";

    cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);

    while (!ctx_->stop.load(std::memory_order_acquire)) {

        cv::Mat frame;
        auto t_start = std::chrono::steady_clock::now();

        if (ctx_->display_queue.tryConsume(frame)) {

            // 1. 绘制语义分割覆盖
            auto seg = ctx_->seg_result.get();
            if (seg && !seg->mask.empty()) {
                drawSegmentation(frame, *seg);
            }

            // 2. 绘制目标检测框
            auto det = ctx_->det_result.get();
            if (det && !det->boxes.empty()) {
                drawDetections(frame, *det);
            }

            cv::imshow(kWindowName, frame);

            auto t_end = std::chrono::steady_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_end - t_start).count();
            ctx_->perf_stats.record(SLOT_DISPLAY, elapsed_us);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }

        int key = cv::waitKey(1);
        if (key == 27) {
            ctx_->stop.store(true, std::memory_order_release);
        }
    }

    cv::destroyWindow(kWindowName);
    std::cout << "[DisplayTask] 已退出" << std::endl;
}
