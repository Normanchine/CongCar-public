#pragma once

#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <iostream>
#include "SharedContext.h"
#include "ThreadTaskFactory.h"

class Car {
public:
    Car() = default;
    ~Car() { stop(); }

    Car(const Car&) = delete;
    Car& operator=(const Car&) = delete;

    // 解析命令行参数 → 加载JSON配置 → 初始化线程池（调试模式）
    bool init(int argc, char* argv[]);

    // 根据config依次启动各工作线程
    void start();

    // 通知所有线程退出 → join全部线程 → 销毁线程池
    void stop();

    // 仅设置退出标志（信号安全，Ctrl+C 处理用）
    void signalStop() { ctx_.stop.store(true, std::memory_order_release); }

    // 阻塞等待所有线程结束
    void wait();

private:
    // 启动单个线程任务
    void launchTask(const std::string& name);

    SharedContext ctx_;
    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<IThreadTask>> tasks_;
    std::unique_ptr<ThreadPool> thread_pool_;   // 调试模式持有，比赛模式为nullptr
    std::thread perf_timer_;                    // T1定时器：每5秒提交性能统计到线程池
    bool initialized_ = false;
};
