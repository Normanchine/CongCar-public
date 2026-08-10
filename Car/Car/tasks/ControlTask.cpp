#include "ControlTask.h"
#include "../common/PerfStats.h"
#include "../hal/uart/UartIO.h"
#include "../thread/ThreadTaskFactory.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

REGISTER_THREAD_TASK(ControlTask)

ControlTask::ControlTask(SharedContext* ctx) : IThreadTask(ctx) {}
ControlTask::~ControlTask() = default;

void ControlTask::run() {
    UartIO uart;
    const auto& uc = ctx_->config.uart;

    // ---- 0. 远程发车：终端输入监听（一行=一次指令，等价下位机按键） ----
    // 发车阶段首次输入=启动；主循环输入=启动/急停切换（与物理按键合并）
    std::cout << "[ControlTask] 远程发车已启用：终端输入任意字符+回车 = 发车/急停" << std::endl;
    {
        std::thread stdin_thread([this]() {
            char line[64];
            while (!ctx_->stop.load(std::memory_order_acquire)) {
                if (!std::fgets(line, sizeof(line), stdin)) {
                    // 无终端/EOF，避免忙等（后台运行时 stdin 直接 EOF）
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                // 空行/纯回车忽略；有内容的一行 = 一次发车指令
                bool has_cmd = false;
                for (const char* p = line; *p; ++p) {
                    if (*p != '\n' && *p != '\r' && *p != ' ' && *p != '\t') {
                        has_cmd = true;
                        break;
                    }
                }
                if (!has_cmd) continue;
                std::cout << "[ControlTask] 终端指令: " << line;
                ctx_->remote_keypress.store(true, std::memory_order_release);
            }
        });
        stdin_thread.detach();   // fgets 阻塞无法 join，立即分离；进程退出即结束
    }

    // ---- 1. 连接串口（支持重试） ----
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        if (uart.open(uc.port, uc.baud)) {
            std::cout << "[ControlTask] 串口已打开 " << uc.port << " @" << uc.baud << std::endl;
            break;
        }
        std::cerr << "[ControlTask] 串口打开失败，5秒后重试..." << std::endl;
        for (int i = 0; i < 50; ++i) {
            if (ctx_->stop.load(std::memory_order_acquire)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    if (ctx_->stop.load(std::memory_order_acquire)) return;

    uart.startReceive();
    ctx_->uart_io = &uart;  // 场景可通过ctx直接发串口指令

    // ---- 2. 等待下位机按键启动 ----
    std::cout << "[ControlTask] 等待按键启动..." << std::endl;
    bool running = false;
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        if (uart.consumeKeypress() || ctx_->remote_keypress.exchange(false)) {
            running = true;
            std::cout << "[ControlTask] 按键/终端发车启动" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (ctx_->stop.load(std::memory_order_acquire)) return;

    // ---- 3. 主控制循环（按键只控制串口下发，不影响场景执行） ----
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        // 3a. 等待新分割结果（阻塞），超时 50ms 兜底
        {
            std::unique_lock<std::mutex> lock(ctx_->seg_cv_mtx_);
            ctx_->seg_cv_.wait_for(lock, std::chrono::milliseconds(50));
        }
        if (ctx_->stop.load(std::memory_order_acquire)) break;

        auto t_start = std::chrono::steady_clock::now();

        // 3b. 执行当前场景（不管按键状态，场景始终运行）
        ctx_->scene_manager.run(ctx_);

        // 3c. 按键/终端指令切换：只控制是否下发串口指令
        if (uart.consumeKeypress() || ctx_->remote_keypress.exchange(false)) {
            running = !running;
            std::cout << "[ControlTask] 通信: " << (running ? "开启" : "关闭") << std::endl;
        }

        // 3d. 下发控制（running=false 时发空指令，车不动）
        if (running) {
            float speed = ctx_->target_speed.load(std::memory_order_acquire);
            float pwm   = ctx_->steering_pwm.load(std::memory_order_acquire);
            uart.sendControl(speed, static_cast<uint16_t>(pwm));
        } else {
            uart.sendControl(0.0f, static_cast<uint16_t>(uc.pwm_neutral));
        }

        // 3e. 性能统计
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        ctx_->perf_stats.record(SLOT_CONTROL, elapsed_us);
    }

    // 退出前发送停止指令
    uart.sendControl(0.0f, static_cast<uint16_t>(uc.pwm_neutral));
    std::cout << "[ControlTask] 已发送停止指令" << std::endl;

    uart.stopReceive();
    uart.close();
    std::cout << "[ControlTask] 已退出" << std::endl;
}
