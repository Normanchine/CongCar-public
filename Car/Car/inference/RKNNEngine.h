#pragma once

#include "../hal/rknn/rknn_api.h"
#include <opencv2/core/mat.hpp>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>

// 单NPU上下文 + 内部worker线程，提供异步 submit/get 流水线
class RKNNEngine {
public:
    RKNNEngine() = default;
    ~RKNNEngine() { release(); }

    RKNNEngine(const RKNNEngine&) = delete;
    RKNNEngine& operator=(const RKNNEngine&) = delete;

    bool load(const std::string& model_path, int core_id,
              rknn_tensor_type input_type = RKNN_TENSOR_UINT8);

    // 返回类型: pair<推理结果, 纯rknn_run耗时(微秒)>
    using Result = std::pair<std::vector<cv::Mat>, int64_t>;

    // 异步提交，返回 future（取走输入所有权）
    std::future<Result> submit(cv::Mat input);

    // 同步推理
    Result infer(const cv::Mat& input) {
        auto fut = submit(input);
        return fut.get();
    }

    void release();
    bool isReady() const { return ready_; }

    int inputWidth()  const { return input_w_; }
    int inputHeight() const { return input_h_; }

private:
    struct Task {
        cv::Mat input;
        std::promise<Result> promise;
    };

    void workerLoop();

    bool ready_ = false;
    int  input_w_ = 0, input_h_ = 0;
    uint32_t n_outputs_ = 0;
    rknn_tensor_type input_type_ = RKNN_TENSOR_UINT8;
    std::vector<rknn_tensor_attr> output_attrs_;

    rknn_context ctx_ = 0;
    std::thread worker_;

    std::queue<Task>    task_queue_;
    std::mutex          queue_mtx_;
    std::condition_variable queue_cv_;
    std::atomic<bool>   stop_{false};
};
