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

REGISTER_SCENE(TrafficSignScene)

// ============================================================================
// Follower 子类方法
// ============================================================================

void TrafficSignScene::Follower::onCenterline(const cv::Mat& mask,
                                              std::vector<cv::Point2f>& centers,
                                              SharedContext* ctx) {
    // 右转锁中（决策右转）：沿右边界（extractBoundary 取每行最右=岔路外沿），走右岔路。
    // 非锁中（APPROACH 接近路牌 / 直走）：中线巡线（等宽处理后取左右中点）。
    bool lock = ctx->right_turn_lock_seconds.load(std::memory_order_acquire) > 0.0f;
    setFollowEdge(lock ? 1 : 0);
    pathplan::extractBoundary(mask, centers, lock ? 1 : 0);
    pathplan::smoothCenterline(centers);
}

void TrafficSignScene::Follower::onFork(cv::Mat& wide_mask, SharedContext* ctx) {
    const auto& fk = ctx->config.control.fork;
    pathplan::handleFork(wide_mask, fork_state_,
                         fork_side, fk.confirm_rows,
                         fk.persist_frames, fk.width_threshold, 1);
}

bool TrafficSignScene::Follower::blindTurnRight(int frames, float speed, SharedContext* ctx) {
    const auto& ctrl = ctx->config.control;

    // 首次调用初始化盲开帧数
    if (turn_frames_left_ <= 0) {
        turn_frames_left_ = frames;
    }

    // 全程右满舵：steering_dir 处理 AR 镜像（正常=右偏到 pwm_max，镜像=左偏到 pwm_min）
    float pwm = (ctx->steering_dir > 0) ? ctrl.pwm_max : ctrl.pwm_min;
    last_pwm_ = pwm;
    ctx->steering_pwm.store(pwm);

    // 盲开固定速度（场景参数 turn_blind_speed）
    last_speed_ = speed;
    ctx->target_speed.store(last_speed_);

    turn_frames_left_--;
    return turn_frames_left_ > 0;
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
    if (best && best_score > 0.4f) {
        stable_count_++;
        if (stable_count_ >= vlm_cfg.stable_frames) {
            delay_counter_++;
            if (delay_counter_ >= vlm_cfg.delay_frames) {
                // 右转锁锁定中：决策锁住，直接复用右转（不发 VLM、不重新判断）
                if (ctx->right_turn_lock_seconds.load(std::memory_order_acquire) > 0.0f) {
                    cached_fork_side_ = 1;
                    vlm_requested_ = false;
                    ctx->vlm_decision.store(1, std::memory_order_release);
                    ctx->vlm_state.store(2, std::memory_order_release);
                    std::cout << "[TrafficSignScene] 右转锁中，复用右转，跳过 VLM" << std::endl;
                    beginBackup(ctx);  // 停车 + 倒车，倒完直接 DRIVE（右转盲开）
                    return;
                }

                // 全局决策已设置（全赛只判断一次）：跳过 VLM，直接复用缓存决策
                int global_decision = ctx->traffic_decision.load(std::memory_order_acquire);

                if (global_decision < 0) {
                    // ---- 首次：未决策，取工作帧发起 VLM 请求 ----
                    cv::Mat work_frame;
                    if (ctx->work_frame_queue.tryConsume(work_frame)) {
                        auto jpeg_bytes = VlmClient::prepareSignImage(
                            work_frame, best->x1, best->y1, best->x2, best->y2, vlm_cfg);

                        std::cout << "[TrafficSignScene] 发起 VLM 请求 score="
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

                        vlm_requested_ = true;
                        beginBackup(ctx);  // 停车 + 倒车，倒完进 WAIT_VLM 等 VLM 决策
                        return;
                    }
                } else {
                    // ---- 复用全局决策：不发 VLM，直接按缓存决策走 ----
                    cached_fork_side_ = global_decision;
                    vlm_requested_ = false;
                    ctx->vlm_decision.store(global_decision, std::memory_order_release);
                    ctx->vlm_state.store(2, std::memory_order_release);
                    std::cout << "[TrafficSignScene] 复用全局决策: "
                              << (global_decision == 0 ? "直走" : "右转")
                              << "，跳过 VLM" << std::endl;
                    beginBackup(ctx);  // 停车 + 倒车，倒完直接 DRIVE
                    return;
                }
            }
            stable_count_ = 0;
        }
    } else {
        stable_count_ = 0;
        delay_counter_ = 0;
    }

    // 未触发 VLM：正常巡线（进入场景默认右边界，路牌在右侧，保证看到完整路牌）
    follower_.fork_side = 1;
    follower_.process(seg_mask, ctx);
}

// ============================================================================
// 阶段1.5：停车 + 倒车退回（惯性冲过头，倒车退到合适位置）
// ============================================================================

void TrafficSignScene::beginBackup(SharedContext* ctx) {
    const float pwm_neutral = ctx->config.uart.pwm_neutral;

    // 先停车（已停稳，再倒车）
    ctx->target_speed.store(0.0f, std::memory_order_release);
    ctx->steering_pwm.store(pwm_neutral, std::memory_order_release);
    ctx->uart_io->sendControl(0.0f, static_cast<uint16_t>(pwm_neutral));

    // 进入倒车阶段
    const auto& tscfg = ctx->config.control.scene.at("TrafficSignScene");
    backup_frames_left_ = tscfg.backup_frames;
    vlm_state_ = VlmState::BACKUP;
    ctx->scene_manager.hold();  // 等待决策期间阻止场景切换（防 DetTask 切走）
    delay_counter_ = 0;
    stable_count_ = 0;
}

void TrafficSignScene::handleBackup(SharedContext* ctx) {
    const auto& tscfg = ctx->config.control.scene.at("TrafficSignScene");
    const float pwm_neutral = ctx->config.uart.pwm_neutral;

    // 倒车：负速度（后退）+ 舵回正
    ctx->target_speed.store(tscfg.backup_speed, std::memory_order_release);
    ctx->steering_pwm.store(pwm_neutral, std::memory_order_release);
    ctx->uart_io->sendControl(tscfg.backup_speed, static_cast<uint16_t>(pwm_neutral));

    backup_frames_left_--;
    if (backup_frames_left_ <= 0) {
        // 倒车结束 → 停车
        ctx->target_speed.store(0.0f, std::memory_order_release);
        ctx->steering_pwm.store(pwm_neutral, std::memory_order_release);
        ctx->uart_io->sendControl(0.0f, static_cast<uint16_t>(pwm_neutral));

        if (vlm_requested_) {
            // 首次：发起了 VLM → 进 WAIT_VLM 等决策
            vlm_state_ = VlmState::WAIT_VLM;
            std::cout << "[TrafficSignScene] 倒车完成，停车等待 VLM 决策" << std::endl;
        } else {
            // 复用全局决策：未发 VLM → 直接进 DRIVE 按缓存决策走
            vlm_state_ = VlmState::DRIVE;
            std::cout << "[TrafficSignScene] 倒车完成，按全局决策行驶" << std::endl;
        }
    }
}

// ============================================================================
// 阶段2：停车等待 VLM 决策返回
// ============================================================================

void TrafficSignScene::handleWaitVlm(SharedContext* ctx) {
    if (!vlm_result_) return;

    int r = vlm_result_->load(std::memory_order_acquire);
    if (r >= 0) {
        cached_fork_side_ = r;   // 本次进场景按 VLM 结果走
        // 写入全局的应是"下一次"的方向：全赛只有一圈右转，本次右转后下次就是直走。
        // 因此 traffic_decision 存 VLM 结果的反值，下次进场景直接读全局当 cached_fork_side_。
        int next = (r == 1) ? 0 : 1;
        ctx->traffic_decision.store(next, std::memory_order_release);

        vlm_state_ = VlmState::DRIVE;
        // 不立即 release：右转盲开期间保持 hold 阻止切换；直走/盲开结束时由 handleDrive release

        auto now = std::chrono::steady_clock::now();
        auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - vlm_request_start_).count();
        ctx->vlm_latency_ms.store(latency_ms, std::memory_order_release);
        ctx->vlm_decision.store(r, std::memory_order_release);
        ctx->vlm_state.store(2, std::memory_order_release);

        std::cout << "[TrafficSignScene] VLM 决策: "
                  << (cached_fork_side_ == 0 ? "直走" : "右转")
                  << " (耗时 " << latency_ms << "ms)；写入全局下次= "
                  << (next == 0 ? "直走" : "右转") << std::endl;
    }
    // 未完成 → 继续停车等待（什么都不做）
}

// ============================================================================
// 阶段3：按 VLM 决策执行巡线
// ============================================================================

void TrafficSignScene::handleDrive(SharedContext* ctx, const cv::Mat& seg_mask) {
    // 全程宽鸟瞰由 PathFollower 内部统一处理，无需临时改 bird_width
    follower_.fork_side = cached_fork_side_;

    if (cached_fork_side_ == 1 && !follower_.turn_blind_done_) {
        // 右转决策：岔路被等宽裁剪走不了巡线，改用盲开右转
        const auto& tscfg = ctx->config.control.scene.at("TrafficSignScene");
        if (follower_.blindTurnRight(tscfg.turn_blind_frames, tscfg.turn_blind_speed, ctx))
            return;  // 盲开中（hold 阻止切换）
        follower_.turn_blind_done_ = true;   // 盲开结束，标记完成

        // 盲开结束后开启右转锁：决策锁住 + 跳过等宽，right_turn_lock_seconds 秒。
        // 锁的全部时长用于"进入岔路的右边界巡线"（盲开本身满舵硬走，不需锁）。
        // ControlTask 每帧递减，到 0 自动解锁；退出场景后仍生效。
        ctx->right_turn_lock_seconds.store(tscfg.right_turn_lock_seconds,
                                           std::memory_order_release);
        std::cout << "[TrafficSignScene] 右转锁开启 " << tscfg.right_turn_lock_seconds
                  << "s（决策锁住+跳过等宽）" << std::endl;

        ctx->scene_manager.release();        // 恢复允许切换
    } else if (cached_fork_side_ == 0) {
        ctx->scene_manager.release();        // 直走：恢复允许切换，正常巡线
    }

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
    case VlmState::BACKUP:   handleBackup(ctx);                break;
    case VlmState::WAIT_VLM: handleWaitVlm(ctx);               break;
    case VlmState::DRIVE:    handleDrive(ctx, seg->mask);       break;
    }
}
