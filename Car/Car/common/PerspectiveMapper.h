#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

// 透视变换工具：初始化后调用 warp() 将任意图像转为鸟瞰图
class PerspectiveMapper {
public:
    void init(const float src_quad[8], int bird_w, int bird_h, float scale_w = 1.0f) {
        cv::Point2f src[4], dst[4];
        for (int i = 0; i < 4; ++i) {
            src[i] = cv::Point2f(src_quad[i * 2], src_quad[i * 2 + 1]);
        }

        if (scale_w <= 1.0f) {
            // 无扩展：直接使用标定参数
            dst[0] = cv::Point2f(0, 0);
            dst[1] = cv::Point2f(static_cast<float>(bird_w - 1), 0);
            dst[2] = cv::Point2f(static_cast<float>(bird_w - 1), static_cast<float>(bird_h - 1));
            dst[3] = cv::Point2f(0, static_cast<float>(bird_h - 1));
            M_ = cv::getPerspectiveTransform(src, dst);
            bird_w_ = bird_w;
            bird_h_ = bird_h;
        } else {
            // 宽度扩展：单应矩阵地面外推（ppm不变，物理视野等比扩大）
            dst[0] = cv::Point2f(0, 0);
            dst[1] = cv::Point2f(static_cast<float>(bird_w - 1), 0);
            dst[2] = cv::Point2f(static_cast<float>(bird_w - 1), static_cast<float>(bird_h - 1));
            dst[3] = cv::Point2f(0, static_cast<float>(bird_h - 1));
            cv::Mat M_orig = cv::getPerspectiveTransform(src, dst);
            cv::Mat M_inv_tmp = M_orig.inv();

            int new_w = static_cast<int>(bird_w * scale_w);
            float offset = (new_w - bird_w) / 2.0f;

            // 扩展后BEV四角在原BEV坐标系中的坐标
            std::vector<cv::Point2f> old_bev_corners = {
                {-offset, 0.0f},
                {bird_w - 1.0f + offset, 0.0f},
                {bird_w - 1.0f + offset, static_cast<float>(bird_h - 1)},
                {-offset, static_cast<float>(bird_h - 1)}
            };

            // 反算回图像坐标，得到新的源四角
            std::vector<cv::Point2f> new_src;
            cv::perspectiveTransform(old_bev_corners, new_src, M_inv_tmp);

            // 扩展后的目标四角
            cv::Point2f new_dst[4] = {
                {0.0f, 0.0f},
                {static_cast<float>(new_w - 1), 0.0f},
                {static_cast<float>(new_w - 1), static_cast<float>(bird_h - 1)},
                {0.0f, static_cast<float>(bird_h - 1)}
            };

            M_ = cv::getPerspectiveTransform(new_src.data(), new_dst);
            bird_w_ = new_w;
            bird_h_ = bird_h;
        }
        M_inv_ = M_.inv();
        ready_ = true;
    }

    cv::Mat warp(const cv::Mat& src) const {
        cv::Mat dst;
        cv::warpPerspective(src, dst, M_, cv::Size(bird_w_, bird_h_));
        return dst;
    }

    bool isReady() const { return ready_; }
    int  birdW()   const { return bird_w_; }
    int  birdH()   const { return bird_h_; }

    // 图像坐标 → 鸟瞰坐标（单点变换，微秒级）
    cv::Point2f toBird(const cv::Point2f& p) const {
        std::vector<cv::Point2f> src = {p};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, M_);
        return dst[0];
    }

    // 鸟瞰图 → 图像坐标（整图逆透视变换，PersonScene 弧线回投用）
    cv::Mat warpInverse(const cv::Mat& bird, int dst_w, int dst_h) const {
        cv::Mat dst;
        cv::warpPerspective(bird, dst, M_inv_, cv::Size(dst_w, dst_h));
        return dst;
    }

private:
    cv::Mat M_;
    cv::Mat M_inv_;
    int bird_w_ = 0;
    int bird_h_ = 0;
    bool ready_ = false;
};
