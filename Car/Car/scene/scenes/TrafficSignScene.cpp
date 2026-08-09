#include "TrafficSignScene.h"
#include "../../common/VlmClient.h"
#include "../../hal/uart/UartIO.h"
#include "../../thread/SharedContext.h"
#include "../framework/SceneFactory.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <thread>

//REGISTER_SCENE(TrafficSignScene)

// ============================================================================
// Follower 子类方法
// ============================================================================

void TrafficSignScene::Follower::onCenterline(const cv::Mat& mask,
                                              std::vector<cv::Point2f>& centers,
                                              SharedContext* /*ctx*/) {
    // 直走(0)→沿左边界(-1)；右转(1)→沿右边界(+1)
    pathplan::extractBoundary(mask, centers, (fork_side == 1) ? 1 : -1);
    pathplan::smoothCenterline(centers);
}

void TrafficSignScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    const auto& fk = ctx->config.control.fork;
    pathplan::handleFork(wide_mask, fork_state_,
                         fork_side, fk.confirm_rows,
                         fk.persist_frames, fk.width_threshold, 1);
}

// ============================================================================
// 阶段1：正常巡线 + 检测交通标志
// ============================================================================

void TrafficSignScene::handleApproach(SharedContext* ctx, const cv::Mat& seg_mask) {
    ctx->vlm_state.store(0, std::memory_order_release);

    const auto& vlm_cfg = ctx->config.ai.vlm;
    const auto& det_cfg = ctx->config.ai.detection;

    // 查找 sign class_id（一次）
    if (traffic_sign_class_id_ < 0) {
        const auto& classes = det_cfg.classes;
        for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
            if (classes[i] == "sign") {
                traffic_sign_class_id_ = i;
                std::cout << "[TrafficSignScene] 找到 sign class_id=" << i << std::endl;
                break;
            }
        }
        if (traffic_sign_class_id_ < 0) {
            std::cerr << "[TrafficSignScene] 未找到 sign 类别！classes=[";
            for (size_t i = 0; i < classes.size(); ++i) {
                if (i > 0) std::cerr << ",";
                std::cerr << classes[i];
            }
            std::cerr << "]" << std::endl;
        }
    }

    // 找置信度最高的 sign 框
    auto det = ctx->det_result.get();
    const DetectionBox* best = nullptr;
    float best_score = 0.0f;
    if (det && traffic_sign_class_id_ >= 0) {
        for (const auto& b : det->boxes) {
            if (b.class_id == traffic_sign_class_id_ && b.confidence > best_score) {
                best_score = b.confidence;
                best = &b;
            }
        }
    }

    // 稳定计数 + 延时 → 发起 VLM 请求
    const uint16_t pwm_neutral = static_cast<uint16_t>(ctx->config.uart.pwm_neutral);

    if (best && best_score > 0.4f) {
        stable_count_++;
        if (stable_count_ >= vlm_cfg.stable_frames) {
            delay_counter_++;
            if (delay_counter_ >= vlm_cfg.delay_frames) {
                cv::Mat work_frame;
                if (ctx->work_frame_queue.tryConsume(work_frame)) {
                    auto jpeg_bytes = VlmClient::prepareSignImage(
                        work_frame, best->x1, best->y1, best->x2, best->y2, vlm_cfg);

                    // 先停车，再发 VLM 请求
                    ctx->uart_io->sendControl(0.0f, pwm_neutral);
                    ctx->target_speed.store(0.0f, std::memory_order_release);
                    ctx->steering_pwm.store(static_cast<float>(pwm_neutral), std::memory_order_release);

                    std::cout << "[TrafficSignScene] 已停车，发起 VLM 请求 score="
                              << best_score << " size=" << jpeg_bytes.size() / 1024 << "KB"
                              << std::endl;

                    // 调试：异步保存 VLM 输入图像
                    if (ctx->thread_pool) {
                        auto jpeg_copy = jpeg_bytes;
                        ctx->thread_pool->enqueueWithPriority(1, [jpeg = std::move(jpeg_copy)]() {
                            mkdir("./debug_vlm", 0755);
                            auto now = std::chrono::system_clock::now();
                            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now.time_since_epoch()).count();
                            std::string path = "./debug_vlm/vlm_" + std::to_string(ms) + ".jpg";
                            std::ofstream ofs(path, std::ios::binary);
                            if (ofs) {
                                ofs.write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size());
                                std::cout << "[TrafficSignScene] VLM 图像已保存: " << path << std::endl;
                            }
                        });
                    }

                    vlm_request_start_ = std::chrono::steady_clock::now();
                    ctx->vlm_state.store(1, std::memory_order_release);

                    // detached thread + shared_ptr<atomic> 异步执行 VLM 请求
                    // promise 堆分配：避免超时后栈析构导致内层线程踩野指针
                    vlm_result_ = std::make_shared<std::atomic<int>>(-1);
                    std::thread([jpeg = std::move(jpeg_bytes),
                                 cfg = vlm_cfg,
                                 result = vlm_result_]() {
                        auto p = std::make_shared<std::promise<int>>();
                        auto f = p->get_future();

                        std::thread([p, jpeg = std::move(jpeg), cfg]() {
                            try {
                                p->set_value(VlmClient::callVlmApi(jpeg, cfg));
                            } catch (const std::exception& e) {
                                std::cerr << "[TrafficSignScene] VLM 异常: "
                                          << e.what() << "，兜底直走" << std::endl;
                                try { p->set_value(0); } catch (...) {}
                            } catch (...) {
                                std::cerr << "[TrafficSignScene] VLM 未知异常，兜底直走" << std::endl;
                                try { p->set_value(0); } catch (...) {}
                            }
                        }).detach();

                        auto deadline_ms = std::chrono::milliseconds(
                            static_cast<long>(cfg.deadline_seconds * 1000));
                        if (f.wait_for(deadline_ms) == std::future_status::ready) {
                            result->store(f.get(), std::memory_order_release);
                        } else {
                            std::cerr << "[TrafficSignScene] VLM 超时 ("
                                      << cfg.deadline_seconds << "s)，兜底直走" << std::endl;
                            result->store(0, std::memory_order_release);
                        }
                    }).detach();

                    vlm_state_ = VlmState::WAIT_VLM;
                    delay_counter_ = 0;
                    stable_count_ = 0;
                    return;  // 不巡线，等 WAIT_VLM 接管
                }
            }
            stable_count_ = 0;
        }
    } else {
        stable_count_ = 0;
        delay_counter_ = 0;
    }

    // 未触发 VLM：正常巡线
    follower_.fork_side = 0;
    follower_.process(seg_mask, ctx);
}

// ============================================================================
// 阶段2：停车等待 VLM 决策返回
// ============================================================================

void TrafficSignScene::handleWaitVlm(SharedContext* ctx) {
    if (!vlm_result_) return;

    int r = vlm_result_->load(std::memory_order_acquire);
    if (r >= 0) {
        cached_fork_side_ = r;
        vlm_state_ = VlmState::DRIVE;

        auto now = std::chrono::steady_clock::now();
        auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - vlm_request_start_).count();
        ctx->vlm_latency_ms.store(latency_ms, std::memory_order_release);
        ctx->vlm_decision.store(r, std::memory_order_release);
        ctx->vlm_state.store(2, std::memory_order_release);

        std::cout << "[TrafficSignScene] VLM 决策: "
                  << (cached_fork_side_ == 0 ? "直走" : "右转")
                  << " (耗时 " << latency_ms << "ms)" << std::endl;
    }
    // 未完成 → 继续停车等待（什么都不做）
}

// ============================================================================
// 阶段3：按 VLM 决策执行巡线
// ============================================================================

void TrafficSignScene::handleDrive(SharedContext* ctx, const cv::Mat& seg_mask) {
    // 全程宽鸟瞰由 PathFollower 内部统一处理，无需临时改 bird_width
    follower_.fork_side = cached_fork_side_;
    follower_.process(seg_mask, ctx);
}

// ============================================================================
// 主循环
// ============================================================================

void TrafficSignScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;

    switch (vlm_state_) {
    case VlmState::APPROACH: handleApproach(ctx, seg->mask);  break;
    case VlmState::WAIT_VLM: handleWaitVlm(ctx);               break;
    case VlmState::DRIVE:    handleDrive(ctx, seg->mask);       break;
    }
}
