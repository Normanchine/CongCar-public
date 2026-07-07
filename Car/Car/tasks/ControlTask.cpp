#include "ControlTask.h"
#include "../common/PerfStats.h"
#include "../hal/uart/UartIO.h"
#include "../thread/ThreadTaskFactory.h"
#include <chrono>
#include <iostream>
#include <thread>

REGISTER_THREAD_TASK(ControlTask)

ControlTask::ControlTask(SharedContext* ctx) : IThreadTask(ctx) {}
ControlTask::~ControlTask() = default;

void ControlTask::run() {
    UartIO uart;
    const auto& uc = ctx_->config.uart;

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
    std::cout << "[ControlTask] 等待按键启动..." << std::endl;

    // ---- 2. 等待下位机按键信号 ----
    bool running = false;
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        if (uart.consumeKeypress()) {
            running = true;
            std::cout << "[ControlTask] 按键启动" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // ---- 3. 主控制循环 ----
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        // 3a. 等待新分割结果（阻塞），超时 50ms 兜底
        {
            std::unique_lock<std::mutex> lock(ctx_->seg_cv_mtx_);
            ctx_->seg_cv_.wait_for(lock, std::chrono::milliseconds(50));
        }
        if (ctx_->stop.load(std::memory_order_acquire)) break;

        auto t_start = std::chrono::steady_clock::now();

        // 3b. 执行当前场景
        ctx_->scene_manager.run(ctx_);

        // 3c. 急停/启动切换
        if (uart.consumeKeypress()) {
            running = !running;
            //std::cout << (running ? "[ControlTask] 恢复运行\n" : "[ControlTask] 急停\n");
        }

        // 3d. 下发控制信息
        if (running) {
            float speed = ctx_->target_speed.load(std::memory_order_acquire);
            float pwm   = ctx_->steering_pwm.load(std::memory_order_acquire);
            uart.sendControl(speed, static_cast<uint16_t>(pwm));
            //std::cout << "[ControlTask] 发送控制信息: " << speed << " " << pwm << std::endl;
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
