#include "Car.h"
#include "../config/ConfigLoader.h"

bool Car::init(int argc, char* argv[]) {
    // 命令行：./car [config_path]
    std::string config_path = "/home/orangepi/Desktop/new/Car/config/default.json";
    if (argc > 1) {
        config_path = argv[1];
    }

    if (!loadConfig(config_path, ctx_.config)) {
        std::cerr << "[Car] 配置加载失败，使用默认值" << std::endl;
    }

    // 调试模式下创建线程池（共2个工作线程）
    if (ctx_.config.debug_mode) {
        thread_pool_ = std::make_unique<ThreadPool>(2);
        ctx_.thread_pool = thread_pool_.get();
    }

    // 转向方向（AR图像左右镜像时取反）
    ctx_.steering_dir = ctx_.config.control.steering_invert ? -1.0f : 1.0f;

    // 全局沿边方向初始值（场景切换继承用，CarScene 会改写）
    ctx_.follow_edge_global = ctx_.config.control.default_follow_edge;

    // 初始化透视变换（若无标定数据则不生效）
    const auto& c = ctx_.config.calib;
    ctx_.perspective.init(c.src_quad, c.bird_width, c.bird_height, c.bird_width_scale);

    initialized_ = true;
    return true;
}

void Car::launchTask(const std::string& name) {
    auto task = ThreadTaskFactory::instance().create(name, &ctx_);
    if (!task) {
        std::cerr << "[Car] 未注册的线程任务: " << name << std::endl;
        return;
    }

    tasks_.push_back(std::move(task));
    IThreadTask* ptr = tasks_.back().get();
    threads_.emplace_back([ptr] { ptr->run(); });
}

void Car::start() {
    if (!initialized_) return;

    // 按顺序启动各线程（采集 → 语义分割 → 目标检测 → 控制）
    launchTask("CaptureTask");

    launchTask("SegmentationTask");

    launchTask("DetectionTask");

    launchTask("ControlTask");

    // 仅在调试模式下启动显示线程
    if (ctx_.config.debug_mode) {
        launchTask("DisplayTask");
    }

    // T1 性能统计定时器：调试模式下每5秒直接打印报告
    if (ctx_.config.debug_mode) {
        perf_timer_ = std::thread([this] {
            while (!ctx_.stop.load(std::memory_order_acquire)) {
                for (int i = 0; i < 50; ++i) {
                    if (ctx_.stop.load(std::memory_order_acquire)) return;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                ctx_.perf_stats.report();
            }
        });
    }
}

void Car::stop() {
    // 1. 通知所有工作线程退出
    ctx_.stop.store(true, std::memory_order_release);

    // 2. 等待 T1 定时器线程退出
    if (perf_timer_.joinable())
        perf_timer_.join();

    // 3. 等待全部工作线程结束
    for (auto& t : threads_) {
        if (t.joinable())
            t.join();
    }
    threads_.clear();
    tasks_.clear();

    // 4. 销毁线程池（工作线程已退出，不再有任务提交）
    thread_pool_.reset();
    ctx_.thread_pool = nullptr;
}

void Car::wait() {
    for (auto& t : threads_) {
        if (t.joinable())
            t.join();
    }
}
