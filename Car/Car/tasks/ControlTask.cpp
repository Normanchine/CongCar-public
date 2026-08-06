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

    // ---- 2. 等待下位机按键启动 ----
    std::cout << "[ControlTask] 等待按键启动..." << std::endl;
    bool running = false;
    while (!ctx_->stop.load(std::memory_order_acquire)) {
        if (uart.consumeKeypress()) {
            running = true;
            std::cout << "[ControlTask] 按键启动" << std::endl;
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

        // 3b. 同步IMU数据（下位机串口→SharedContext）
        //     先于场景执行，保证场景读到本帧最新航向角（PersonScene 航位推算依赖）
        ctx_->yaw.store(uart.getYaw(), std::memory_order_release);

        // 3c. 执行当前场景（不管按键状态，场景始终运行）
        ctx_->scene_manager.run(ctx_);

        // 3d. 按键切换：只控制是否下发串口指令
        if (uart.consumeKeypress()) {
            running = !running;
            std::cout << "[ControlTask] 通信: " << (running ? "开启" : "关闭") << std::endl;
        }

        // 3e. 下发控制（running=false 时发空指令，车不动）
        if (running) {
            float speed = ctx_->target_speed.load(std::memory_order_acquire);
            float pwm   = ctx_->steering_pwm.load(std::memory_order_acquire);
            uart.sendControl(speed, static_cast<uint16_t>(pwm));
        } else {
            uart.sendControl(0.0f, static_cast<uint16_t>(uc.pwm_neutral));
        }

        // 3f. 性能统计
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
