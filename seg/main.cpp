#include "../Car/Car/hal/rknn/rknn_api.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// ===================== 配置默认值 =====================
static const char*  DEFAULT_MODEL_PATH = "/home/orangepi/Desktop/new/seg/build/model/pplite_seg.rknn";
static const int    MODEL_W = 256;
static const int    MODEL_H = 128;
static const int    NUM_CLASSES = 2;
static const float  OVERLAY_ALPHA = 0.5f;

// ===================== 模型加载 =====================
static bool load_model(const char* path, rknn_context* ctx,
                       int* out_w, int* out_h, uint32_t* n_outputs,
                       std::vector<rknn_tensor_attr>* output_attrs) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "无法打开模型: " << path << std::endl;
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(size);
    file.read(reinterpret_cast<char*>(model_data.data()), size);
    file.close();

    int ret = rknn_init(ctx, model_data.data(),
                        static_cast<uint32_t>(size), 0, nullptr);
    if (ret != 0) {
        std::cerr << "rknn_init 失败, ret=" << ret << std::endl;
        return false;
    }

    rknn_set_core_mask(*ctx, RKNN_NPU_CORE_2);  // 绑定 Core2，与 Car 程序一致

    rknn_input_output_num io_num;
    rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attr{};
    input_attr.index = 0;
    rknn_query(*ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    *out_w = input_attr.dims[2];
    *out_h = input_attr.dims[1];
    std::cout << "模型输入: " << *out_w << "x" << *out_h << std::endl;

    *n_outputs = io_num.n_output;
    output_attrs->resize(*n_outputs);
    for (uint32_t i = 0; i < *n_outputs; ++i) {
        (*output_attrs)[i].index = i;
        rknn_query(*ctx, RKNN_QUERY_OUTPUT_ATTR,
                   &(*output_attrs)[i], sizeof(rknn_tensor_attr));
    }
    std::cout << "输出数: " << *n_outputs << std::endl;

    return true;
}

// ===================== 预处理 =====================
static cv::Mat preprocess(const cv::Mat& bgr, int w, int h) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(w, h));
    return rgb;  // UINT8 NHWC, 量化模型直传
}

// ===================== 推理 =====================
// 返回纯 rknn_run 耗时(us)，与 Car 程序 RKNNEngine 的计时口径一致
static bool infer(rknn_context ctx, const cv::Mat& input,
                  std::vector<cv::Mat>& outputs, int64_t* run_us,
                  const std::vector<rknn_tensor_attr>& output_attrs,
                  uint32_t n_outputs) {
    rknn_input rk_in{};
    rk_in.index = 0;
    rk_in.buf   = input.data;
    rk_in.size  = static_cast<uint32_t>(input.total() * input.elemSize());
    rk_in.type  = RKNN_TENSOR_UINT8;
    rk_in.fmt   = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(ctx, 1, &rk_in);
    if (ret != 0) {
        std::cerr << "rknn_inputs_set 失败" << std::endl;
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    ret = rknn_run(ctx, nullptr);
    auto t1 = std::chrono::steady_clock::now();

    if (ret != 0) {
        std::cerr << "rknn_run 失败" << std::endl;
        return false;
    }

    std::vector<rknn_output> rk_outs(n_outputs);
    for (uint32_t i = 0; i < n_outputs; ++i) {
        rk_outs[i].index       = i;
        rk_outs[i].want_float  = 1;
        rk_outs[i].is_prealloc = 0;
    }
    ret = rknn_outputs_get(ctx, n_outputs, rk_outs.data(), nullptr);
    if (ret != 0) {
        std::cerr << "rknn_outputs_get 失败" << std::endl;
        return false;
    }

    outputs.resize(n_outputs);
    for (uint32_t i = 0; i < n_outputs; ++i) {
        const auto& attr = output_attrs[i];
        int rows = static_cast<int>(attr.n_dims >= 3 ? attr.dims[2] : 1);
        int cols = static_cast<int>(attr.n_dims >= 4 ? attr.dims[3] : 1);

        if (attr.n_dims == 4) {
            int channels = static_cast<int>(attr.dims[1]);
            outputs[i] = cv::Mat(channels, rows * cols, CV_32FC1,
                                 rk_outs[i].buf).clone();
        } else if (attr.n_dims == 3) {
            outputs[i] = cv::Mat(rows, cols, CV_32FC1,
                                 rk_outs[i].buf).clone();
        } else {
            outputs[i] = cv::Mat(1, static_cast<int>(attr.n_elems),
                                 CV_32FC1, rk_outs[i].buf).clone();
        }
    }

    rknn_outputs_release(ctx, n_outputs, rk_outs.data());

    *run_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return true;
}

// ===================== 后处理 =====================
static cv::Mat postprocess(const std::vector<cv::Mat>& outputs,
                           int orig_w, int orig_h, int mw, int mh) {
    // output shape: (2, mw*mh) CV_32FC1 — ch0=背景, ch1=赛道
    const cv::Mat& out = outputs[0];
    const float* ch0 = out.ptr<float>(0);
    const float* ch1 = out.ptr<float>(1);

    cv::Mat mask(mh, mw, CV_8UC1);
    const int total = mw * mh;
    for (int i = 0; i < total; i++) {
        mask.data[i] = (ch0[i] >= ch1[i]) ? 0 : 1;
    }

    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, cv::Size(orig_w, orig_h),
               0, 0, cv::INTER_NEAREST);
    return mask_resized;
}

// ===================== 可视化 =====================
static cv::Mat visualize(const cv::Mat& bgr, const cv::Mat& mask) {
    cv::Mat frame_rgb;
    cv::cvtColor(bgr, frame_rgb, cv::COLOR_BGR2RGB);

    // 分割掩码图
    cv::Mat colored(mask.size(), CV_8UC3);
    colored.setTo(cv::Scalar(0, 0, 0));          // 背景=黑色
    colored.setTo(cv::Scalar(0, 255, 0), mask);  // 赛道=绿色

    // 叠加图
    cv::Mat overlay = frame_rgb.clone();
    cv::Mat mask_3ch;
    cv::cvtColor(mask, mask_3ch, cv::COLOR_GRAY2BGR);
    // overlay.setTo(cv::Scalar(0, 255, 0), mask_3ch, OVERLAY_ALPHA);
cv::addWeighted(overlay, 1.0 - OVERLAY_ALPHA, colored, OVERLAY_ALPHA, 0, overlay);
    // 三栏拼接
    cv::Mat result;
    cv::hconcat(std::vector<cv::Mat>{frame_rgb, colored, overlay}, result);

    cv::Mat result_bgr;
    cv::cvtColor(result, result_bgr, cv::COLOR_RGB2BGR);
    return result_bgr;
}

// ===================== 主函数 =====================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: seg_test <图片路径> [模型路径]" << std::endl;
        std::cerr << "  seg_test input.jpg" << std::endl;
        std::cerr << "  seg_test input.jpg model/route_dec.rknn" << std::endl;
        return 1;
    }

    const char* image_path = argv[1];
    const char* model_path = (argc > 2) ? argv[2] : DEFAULT_MODEL_PATH;

    auto t_total_start = std::chrono::steady_clock::now();

    // 1. 加载模型
    std::cout << "加载模型: " << model_path << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    rknn_context ctx = 0;
    int mw = 0, mh = 0;
    uint32_t n_outputs = 0;
    std::vector<rknn_tensor_attr> output_attrs;

    if (!load_model(model_path, &ctx, &mw, &mh, &n_outputs, &output_attrs))
        return 1;

    auto t_load = std::chrono::steady_clock::now();

    // 2. 读取图片
    std::cout << "读取图片: " << image_path << std::endl;
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) {
        std::cerr << "无法读取图片: " << image_path << std::endl;
        rknn_destroy(ctx);
        return 1;
    }
    int orig_w = bgr.cols;
    int orig_h = bgr.rows;
    std::cout << "原始尺寸: " << orig_w << "x" << orig_h << std::endl;

    // 3. 预处理
    auto t_pre_start = std::chrono::steady_clock::now();
    cv::Mat input = preprocess(bgr, mw, mh);
    auto t_pre_end = std::chrono::steady_clock::now();

    // 4. 首帧预热 (NPU 内部初始化，不计时)
    std::cout << "预热中..." << std::endl;
    {
        std::vector<cv::Mat> warm_outputs;
        int64_t warm_us = 0;
        infer(ctx, input, warm_outputs, &warm_us, output_attrs, n_outputs);
        std::cout << "  warmup: " << (warm_us / 1000.0f) << " ms" << std::endl;
    }

    // 5. 正式推理
    std::cout << "推理中..." << std::endl;
    int64_t run_us = 0;
    auto t_infer_start = std::chrono::steady_clock::now();
    std::vector<cv::Mat> outputs;
    if (!infer(ctx, input, outputs, &run_us, output_attrs, n_outputs)) {
        rknn_destroy(ctx);
        return 1;
    }
    auto t_infer_end = std::chrono::steady_clock::now();

    // 6. 后处理
    auto t_post_start = std::chrono::steady_clock::now();
    cv::Mat mask = postprocess(outputs, orig_w, orig_h, mw, mh);
    auto t_post_end = std::chrono::steady_clock::now();

    // 分析分割结果
    int track_px = cv::countNonZero(mask);
    int total_px = mask.total();
    std::cout << "赛道像素: " << track_px << "/" << total_px
              << " (" << (100.0f * track_px / total_px) << "%)" << std::endl;

    // 7. 可视化
    auto t_viz_start = std::chrono::steady_clock::now();
    cv::Mat result = visualize(bgr, mask);
    auto t_viz_end = std::chrono::steady_clock::now();

    // 8. 保存
    std::string out_name = std::string(image_path);
    size_t dot = out_name.find_last_of('.');
    out_name = (dot == std::string::npos)
        ? out_name + "_result.png"
        : out_name.substr(0, dot) + "_result.png";

    cv::imwrite(out_name, result);
    std::cout << "结果已保存: " << out_name << std::endl;

    // 9. 释放
    rknn_destroy(ctx);

    auto t_total_end = std::chrono::steady_clock::now();

    // ---- 性能统计 ----
    auto load_ms  = std::chrono::duration<float, std::milli>(t_load - t0).count();
    auto pre_ms   = std::chrono::duration<float, std::milli>(t_pre_end - t_pre_start).count();
    auto infer_ms = std::chrono::duration<float, std::milli>(t_infer_end - t_infer_start).count();
    float run_ms  = run_us / 1000.0f;  // 纯 rknn_run，与 Car 程序口径一致
    float oh_ms   = infer_ms - run_ms; // inputs_set + outputs_get + clone 开销
    auto post_ms  = std::chrono::duration<float, std::milli>(t_post_end - t_post_start).count();
    auto viz_ms   = std::chrono::duration<float, std::milli>(t_viz_end - t_viz_start).count();
    auto total_ms = std::chrono::duration<float, std::milli>(t_total_end - t_total_start).count();

    std::cout << "\n"
              << std::string(60, '=') << "\n"
              << "  Performance Statistics:\n"
              << "  Model Load:    " << std::fixed << std::setprecision(2)
                                      << load_ms << " ms\n"
              << "  Preprocessing:  " << pre_ms  << " ms"
              << " (" << (pre_ms / total_ms * 100) << "%)\n"
              << "  Inference:      " << run_ms  << " ms  (纯 rknn_run)"
              << " (" << (run_ms / total_ms * 100) << "%)\n"
              << "  API 开销:       " << oh_ms   << " ms  (inputs_set+outputs_get+clone)"
              << " (" << (oh_ms / total_ms * 100) << "%)\n"
              << "  Postprocessing: " << post_ms << " ms"
              << " (" << (post_ms / total_ms * 100) << "%)\n"
              << "  Visualization:  " << viz_ms  << " ms"
              << " (" << (viz_ms / total_ms * 100) << "%)\n"
              << "  Total:          " << total_ms << " ms\n"
              << std::string(60, '=') << std::endl;

    return 0;
}
