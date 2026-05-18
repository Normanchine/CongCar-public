#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

// 透视变换工具：初始化后调用 warp() 将任意图像转为鸟瞰图
class PerspectiveMapper {
public:
    void init(const float src_quad[8], int bird_w, int bird_h) {
        cv::Point2f src[4], dst[4];
        for (int i = 0; i < 4; ++i) {
            src[i] = cv::Point2f(src_quad[i * 2], src_quad[i * 2 + 1]);
        }
        dst[0] = cv::Point2f(0, 0);
        dst[1] = cv::Point2f(static_cast<float>(bird_w - 1), 0);
        dst[2] = cv::Point2f(static_cast<float>(bird_w - 1), static_cast<float>(bird_h - 1));
        dst[3] = cv::Point2f(0, static_cast<float>(bird_h - 1));

        M_ = cv::getPerspectiveTransform(src, dst);
        bird_w_ = bird_w;
        bird_h_ = bird_h;
        ready_ = true;
    }

    cv::Mat warp(const cv::Mat& src) const {
        cv::Mat dst;
        cv::warpPerspective(src, dst, M_, cv::Size(bird_w_, bird_h_));
        return dst;
    }

    bool isReady() const { return ready_; }

private:
    cv::Mat M_;
    int bird_w_ = 0;
    int bird_h_ = 0;
    bool ready_ = false;
};
