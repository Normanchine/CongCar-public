#include "RKNNEngine.h"
#include <fstream>
#include <iostream>

bool RKNNEngine::load(const std::string& model_path, int core_id,
                       rknn_tensor_type input_type) {
    input_type_ = input_type;
    release();

    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[RKNNEngine] 无法打开模型: " << model_path << std::endl;
        return false;
    }
    size_t model_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(model_size);
    file.read(reinterpret_cast<char*>(model_data.data()), model_size);
    file.close();

    int ret = rknn_init(&ctx_, model_data.data(),
                        static_cast<uint32_t>(model_size), 0, nullptr);
    if (ret != 0) {
        std::cerr << "[RKNNEngine] rknn_init 失败, ret=" << ret << std::endl;
        return false;
    }

    rknn_core_mask cores[] = {
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2
    };
    rknn_set_core_mask(ctx_, cores[core_id % 3]);

    rknn_input_output_num io_num;
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attr{};
    input_attr.index = 0;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    input_w_ = input_attr.dims[2];
    input_h_ = input_attr.dims[1];

    n_outputs_ = io_num.n_output;
    output_attrs_.resize(n_outputs_);
    for (uint32_t i = 0; i < n_outputs_; ++i) {
        output_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR,
                   &output_attrs_[i], sizeof(rknn_tensor_attr));
    }

    stop_ = false;
    worker_ = std::thread(&RKNNEngine::workerLoop, this);

    ready_ = true;
    std::cout << "[RKNNEngine] 模型已加载: " << model_path
              << " (" << input_w_ << "x" << input_h_ << ")"
              << ", NPU Core " << (core_id % 3)
              << ", 输出数: " << n_outputs_ << std::endl;
    return true;
}

void RKNNEngine::release() {
    stop_ = true;
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (ctx_) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    ready_ = false;
}

std::future<RKNNEngine::Result> RKNNEngine::submit(cv::Mat input) {
    std::promise<Result> promise;
    auto future = promise.get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mtx_);
        task_queue_.push({std::move(input), std::move(promise)});
    }
    queue_cv_.notify_one();
    return future;
}

void RKNNEngine::workerLoop() {
    while (!stop_) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] {
                return stop_ || !task_queue_.empty();
            });
            if (stop_ && task_queue_.empty()) return;
            if (task_queue_.empty()) continue;
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        rknn_input rk_in{};
        rk_in.index = 0;
        rk_in.buf   = const_cast<void*>(static_cast<const void*>(task.input.data));
        rk_in.size  = static_cast<uint32_t>(task.input.total() * task.input.elemSize());
        rk_in.type  = input_type_;
        rk_in.fmt   = RKNN_TENSOR_NHWC;

        int ret = rknn_inputs_set(ctx_, 1, &rk_in);
        if (ret != 0) {
            task.promise.set_value({{}, 0});
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();
        ret = rknn_run(ctx_, nullptr);
        auto t1 = std::chrono::steady_clock::now();

        if (ret != 0) {
            task.promise.set_value({{}, 0});
            continue;
        }

        std::vector<rknn_output> rk_outs(n_outputs_);
        for (uint32_t i = 0; i < n_outputs_; ++i) {
            rk_outs[i].index       = i;
            rk_outs[i].want_float  = 1;
            rk_outs[i].is_prealloc = 0;
        }
        ret = rknn_outputs_get(ctx_, n_outputs_, rk_outs.data(), nullptr);
        if (ret != 0) {
            task.promise.set_value({{}, 0});
            continue;
        }

        std::vector<cv::Mat> outputs(n_outputs_);
        for (uint32_t i = 0; i < n_outputs_; ++i) {
            const auto& attr = output_attrs_[i];
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

        rknn_outputs_release(ctx_, n_outputs_, rk_outs.data());

        auto infer_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t1 - t0).count();

        task.promise.set_value({std::move(outputs), infer_us});
    }
}
