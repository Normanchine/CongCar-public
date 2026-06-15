//
// simble_test — 语义分割 + 路径规划一体化测试程序
// 功能: 共享内存读帧 → NPU语义分割 → 透视变换 → 中线提取 → 纯跟踪 → HTTP推流
// 去除了: 目标检测、上下位机通信、PWM控制、本地显示
// 新增: HTTP MJPEG推流，支持浏览器远程查看
//

#include "rknn_api.h"
#include "PathPlanning.h"
#include "PerspectiveMapper.h"
#include "HttpStreamServer.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

using json = nlohmann::json;

// ============================================================================
// 配置结构体 — 从JSON读取
// ============================================================================
struct SimbleConfig {
    std::string shm_name = "shm_ar_video";
    size_t shm_size = 10 * 1024 * 1024;
    std::string model_path = "/home/orangepi/Desktop/new/seg/build/model/pplite_seg.rknn";
    int segmentation_input_w = 256;
    int segmentation_input_h = 128;
    // 车辆物理参数（路径规划用）
    float car_length = 0.316f;
    float car_cy_ratio = 0.95f;
    float aim_dist_near = 0.35f;
    int track_search_turn_dir = 1;  // 横向赛道入轨转向
    // 标定参数
    float src_quad[8] = {0};
    int bird_width = 640;
    int bird_height = 480;
    float ppm_x = 0.0f;  // X方向像素/米
    float ppm_y = 0.0f;  // Y方向像素/米
    // HTTP推流参数
    int http_port = 5001;           // HTTP服务器端口
    int jpeg_quality = 80;          // JPEG编码质量（1-100）
    bool enable_http_stream = true; // 是否启用HTTP推流
};

// 从JSON文件加载配置
static bool loadConfig(const std::string& path, SimbleConfig& cfg) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[配置] 无法打开配置文件: " << path << std::endl;
        return false;
    }
    try {
        json j = json::parse(f);
        if (j.contains("shm_name")) cfg.shm_name = j["shm_name"];
        if (j.contains("shm_size")) cfg.shm_size = j["shm_size"];
        if (j.contains("model_path")) cfg.model_path = j["model_path"];
        if (j.contains("segmentation_input_w")) cfg.segmentation_input_w = j["segmentation_input_w"];
        if (j.contains("segmentation_input_h")) cfg.segmentation_input_h = j["segmentation_input_h"];
        if (j.contains("car_length")) cfg.car_length = j["car_length"];
        if (j.contains("car_cy_ratio")) cfg.car_cy_ratio = j["car_cy_ratio"];
        if (j.contains("aim_dist_near")) cfg.aim_dist_near = j["aim_dist_near"];
        if (j.contains("track_search_turn_dir")) cfg.track_search_turn_dir = j["track_search_turn_dir"];
        if (j.contains("calib")) {
            auto calib = j["calib"];
            if (calib.contains("src_quad")) {
                auto src = calib["src_quad"];
                for (int i = 0; i < 8 && i < src.size(); ++i) {
                    cfg.src_quad[i] = src[i];
                }
            }
            if (calib.contains("bird_width")) cfg.bird_width = calib["bird_width"];
            if (calib.contains("bird_height")) cfg.bird_height = calib["bird_height"];
            if (calib.contains("ppm_x")) cfg.ppm_x = calib["ppm_x"];
            if (calib.contains("ppm_y")) cfg.ppm_y = calib["ppm_y"];
        }
        // 读取HTTP推流配置
        if (j.contains("http_stream")) {
            auto http = j["http_stream"];
            if (http.contains("port")) cfg.http_port = http["port"];
            if (http.contains("jpeg_quality")) cfg.jpeg_quality = http["jpeg_quality"];
            if (http.contains("enable")) cfg.enable_http_stream = http["enable"];
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[配置] JSON解析错误: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 共享内存读取 — 复用Car程序逻辑
// ============================================================================
constexpr size_t SHM_HEADER_SIZE = 16;  // fid + w + h

struct SharedMemory {
    int fd = -1;
    void* addr = nullptr;
    size_t size = 0;
    std::string name;

    bool open(const std::string& shm_name, size_t shm_size) {
        fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            perror("shm_open");
            return false;
        }
        addr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            perror("mmap");
            ::close(fd);
            fd = -1;
            return false;
        }
        name = shm_name;
        size = shm_size;
        return true;
    }

    void close() {
        if (addr && addr != MAP_FAILED) {
            munmap(addr, size);
            addr = nullptr;
        }
        if (fd != -1) {
            ::close(fd);
            fd = -1;
        }
    }

    bool isValid() const {
        return addr != nullptr && addr != MAP_FAILED;
    }

    uint8_t* getHeader() const { return static_cast<uint8_t*>(addr); }
    uint8_t* getData() const { return static_cast<uint8_t*>(addr) + SHM_HEADER_SIZE; }
};

// ============================================================================
// RKNN模型加载与推理
// ============================================================================
static bool loadRKNModel(const std::string& path, rknn_context* ctx,
                         int* out_w, int* out_h, uint32_t* n_outputs,
                         std::vector<rknn_tensor_attr>* output_attrs) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[模型] 无法打开: " << path << std::endl;
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
        std::cerr << "[模型] rknn_init 失败, ret=" << ret << std::endl;
        return false;
    }

    rknn_set_core_mask(*ctx, RKNN_NPU_CORE_2);

    rknn_input_output_num io_num;
    rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attr{};
    input_attr.index = 0;
    rknn_query(*ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    *out_w = input_attr.dims[2];
    *out_h = input_attr.dims[1];

    *n_outputs = io_num.n_output;
    output_attrs->resize(*n_outputs);
    for (uint32_t i = 0; i < *n_outputs; ++i) {
        (*output_attrs)[i].index = i;
        rknn_query(*ctx, RKNN_QUERY_OUTPUT_ATTR,
                   &(*output_attrs)[i], sizeof(rknn_tensor_attr));
    }

    std::cout << "[模型] 输入尺寸: " << *out_w << "x" << *out_h << std::endl;
    std::cout << "[模型] 输出数目: " << *n_outputs << std::endl;
    return true;
}

static bool runInference(rknn_context ctx, const cv::Mat& input,
                         std::vector<cv::Mat>& outputs, int64_t* infer_us,
                         const std::vector<rknn_tensor_attr>& output_attrs,
                         uint32_t n_outputs) {
    rknn_input rk_in{};
    rk_in.index = 0;
    rk_in.buf = input.data;
    rk_in.size = static_cast<uint32_t>(input.total() * input.elemSize());
    rk_in.type = RKNN_TENSOR_UINT8;
    rk_in.fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(ctx, 1, &rk_in);
    if (ret != 0) {
        std::cerr << "[推理] rknn_inputs_set 失败" << std::endl;
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    ret = rknn_run(ctx, nullptr);
    auto t1 = std::chrono::steady_clock::now();

    if (ret != 0) {
        std::cerr << "[推理] rknn_run 失败" << std::endl;
        return false;
    }

    std::vector<rknn_output> rk_outs(n_outputs);
    for (uint32_t i = 0; i < n_outputs; ++i) {
        rk_outs[i].index = i;
        rk_outs[i].want_float = 1;
        rk_outs[i].is_prealloc = 0;
    }
    ret = rknn_outputs_get(ctx, n_outputs, rk_outs.data(), nullptr);
    if (ret != 0) {
        std::cerr << "[推理] rknn_outputs_get 失败" << std::endl;
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

    *infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return true;
}

static cv::Mat postprocessSeg(const std::vector<cv::Mat>& outputs,
                              int orig_w, int orig_h, int model_w, int model_h) {
    const cv::Mat& out = outputs[0];
    const float* ch0 = out.ptr<float>(0);
    const float* ch1 = out.ptr<float>(1);

    cv::Mat mask(model_h, model_w, CV_8UC1);
    const int total = model_w * model_h;
    for (int i = 0; i < total; ++i) {
        mask.data[i] = (ch0[i] >= ch1[i]) ? 0 : 1;
    }

    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, cv::Size(orig_w, orig_h),
               0, 0, cv::INTER_NEAREST);
    return mask_resized;
}

// ============================================================================
// 可视化绘制 — 复用Car程序DisplayTask逻辑
// ============================================================================
namespace {
constexpr float kSegAlpha = 0.3f;
const cv::Scalar kSegColor(0, 255, 0);
const cv::Scalar kCenterColor(0, 255, 255);
const cv::Scalar kAimColor(0, 0, 255);
const cv::Scalar kCarColor(255, 0, 0);
const cv::Scalar kHudBg(30, 30, 30);
const cv::Scalar kHudText(220, 220, 220);

// 绘制分割掩码叠加
void drawSegOverlay(cv::Mat& frame, const cv::Mat& mask) {
    cv::Mat overlay = cv::Mat::zeros(frame.size(), CV_8UC3);
    overlay.setTo(kSegColor, mask);
    cv::addWeighted(frame, 1.0f - kSegAlpha, overlay, kSegAlpha, 0, frame);
}

// 在鸟瞰图上绘制中线和前视点
void drawPathplan(cv::Mat& bird_vis,
                  const std::vector<cv::Point2f>& centers,
                  const cv::Point2f& aim_pt,
                  float cx, float cy) {
    // 绘制中线（黄色小点）
    for (const auto& pt : centers) {
        cv::circle(bird_vis, cv::Point(static_cast<int>(pt.x),
                                        static_cast<int>(pt.y)),
                   2, kCenterColor, -1);
    }
    // 绘制前视点（红色大圈）
    cv::circle(bird_vis, cv::Point(static_cast<int>(aim_pt.x),
                                    static_cast<int>(aim_pt.y)),
               5, kAimColor, -1);
    // 绘制车底中心（蓝色大圈）
    cv::circle(bird_vis, cv::Point(static_cast<int>(cx),
                                    static_cast<int>(cy)),
               6, kCarColor, 2);
}

// 绘制HUD信息
void drawHUD(cv::Mat& frame, uint64_t frame_id, int n_centers,
             float heading_error, float fps, float seg_ms) {
    const int x = 12;
    int y = 24;
    cv::Rect roi(x - 8, 8, 320, 80);
    cv::Mat bg = frame(roi);
    cv::Mat dark(roi.height, roi.width, CV_8UC3, kHudBg);
    cv::addWeighted(dark, 0.55f, bg, 0.45f, 0, bg);

    std::ostringstream oss;
    oss << "Frame: " << frame_id;
    cv::putText(frame, oss.str(), cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, kHudText, 1);
    y += 22;

    oss.str("");
    oss << "中线点数: " << n_centers;
    if (n_centers == 0) oss << " (丢失赛道)";
    cv::putText(frame, oss.str(), cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, kHudText, 1);
    y += 22;

    oss.str("");
    oss << std::fixed << std::setprecision(3);
    oss << "航向误差: " << heading_error << " rad";
    cv::putText(frame, oss.str(), cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, kHudText, 1);
    y += 22;

    oss.str("");
    oss << std::fixed << std::setprecision(1);
    oss << "FPS: " << fps << "  Seg: " << seg_ms << "ms";
    cv::putText(frame, oss.str(), cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, kHudText, 1);
}
} // anonymous namespace

// ============================================================================
// 全局变量 — 信号处理
// ============================================================================
std::atomic<bool> g_running(true);

// 信号处理函数（Ctrl+C退出）
void signalHandler(int signum) {
    std::cout << "\n\n[信号] 接收到退出信号，正在停止..." << std::endl;
    g_running.store(false);
}

// ============================================================================
// 主函数 — 流程控制
// ============================================================================
int main(int argc, char* argv[]) {
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    const char* config_path = (argc > 1) ? argv[1] : "config.json";
    std::cout << "= simble_test: 语义分割 + 路径规划一体化测试 =" << std::endl;
    std::cout << "配置文件: " << config_path << std::endl;

    // ===================================================
    // 1. 加载配置
    // ===================================================
    SimbleConfig cfg;
    if (!loadConfig(config_path, cfg)) {
        std::cerr << "加载配置失败，使用默认值" << std::endl;
    }

    // 检查标定参数
    bool has_calib = true;
    for (int i = 0; i < 8; ++i) {
        if (cfg.src_quad[i] == 0 && i != 0) {
            has_calib = false;
            break;
        }
    }
    if (!has_calib) {
        std::cerr << "\n[警告] 透视标定参数 src_quad 全零，请先运行 simble_calib 标定" << std::endl;
    }

    std::cout << "\n--- 配置信息 ---" << std::endl;
    std::cout << "共享内存: " << cfg.shm_name << " (" << cfg.shm_size << " bytes)" << std::endl;
    std::cout << "模型路径: " << cfg.model_path << std::endl;
    std::cout << "模型输入: " << cfg.segmentation_input_w << "x" << cfg.segmentation_input_h << std::endl;
    std::cout << "鸟瞰尺寸: " << cfg.bird_width << "x" << cfg.bird_height << std::endl;
    std::cout << "PPM_X (水平): " << cfg.ppm_x << " px/m" << std::endl;
    std::cout << "PPM_Y (垂直): " << cfg.ppm_y << " px/m" << std::endl;
    std::cout << "车参数: 长度=" << cfg.car_length << "m, 车底y比例=" << cfg.car_cy_ratio << std::endl;
    std::cout << "前视距离: " << cfg.aim_dist_near << "m" << std::endl;
    std::cout << "横向赛道转向: " << (cfg.track_search_turn_dir == 1 ? "右" : "左") << std::endl;
    std::cout << "HTTP推流: " << (cfg.enable_http_stream ? "启用" : "禁用") << std::endl;
    if (cfg.enable_http_stream) {
        std::cout << "HTTP端口: " << cfg.http_port << std::endl;
        std::cout << "JPEG质量: " << cfg.jpeg_quality << std::endl;
    }

    // ===================================================
    // 2. 初始化透视变换
    // ===================================================
    PerspectiveMapper perspective;
    perspective.init(cfg.src_quad, cfg.bird_width, cfg.bird_height);
    std::cout << "\n[透视变换] 初始化完成" << std::endl;

    // ===================================================
    // 3. 加载RKNN模型
    // ===================================================
    rknn_context rknn_ctx = 0;
    int model_w = 0, model_h = 0;
    uint32_t n_outputs = 0;
    std::vector<rknn_tensor_attr> output_attrs;
    if (!loadRKNModel(cfg.model_path, &rknn_ctx, &model_w, &model_h, &n_outputs, &output_attrs)) {
        return 1;
    }

    // 预热（NPU初始化）
    {
        cv::Mat warm_input(model_h, model_w, CV_8UC3);
        warm_input.setTo(0);
        std::vector<cv::Mat> warm_outputs;
        int64_t warm_us = 0;
        runInference(rknn_ctx, warm_input, warm_outputs, &warm_us, output_attrs, n_outputs);
        std::cout << "[预热] " << (warm_us / 1000.0f) << " ms" << std::endl;
    }

    // ===================================================
    // 4. 打开共享内存
    // ===================================================
    SharedMemory shm;
    std::cout << "\n[共享内存] 等待连接: " << cfg.shm_name << std::endl;
    while (!shm.isValid()) {
        if (shm.open(cfg.shm_name, cfg.shm_size)) {
            std::cout << "[共享内存] 连接成功" << std::endl;
            break;
        }
        sleep(1);
    }

    // ===================================================
    // 5. 启动HTTP推流服务器
    // ===================================================
    HttpStreamServer* http_server = nullptr;
    if (cfg.enable_http_stream) {
        http_server = new HttpStreamServer(cfg.http_port, cfg.jpeg_quality);
        if (!http_server->start()) {
            std::cerr << "[HTTP] 服务器启动失败" << std::endl;
            delete http_server;
            http_server = nullptr;
        } else {
            std::cout << "\n[HTTP] 浏览器访问地址: http://<开发板IP>:" << cfg.http_port << std::endl;
        }
    }

    // ===================================================
    // 6. 主循环
    // ===================================================
    uint64_t frame_counter = 0;
    uint64_t last_fid = 0;
    float fps = 0.0f;

    std::cout << "\n=== 开始循环，Ctrl+C退出 ===" << std::endl;

    while (g_running.load()) {
        auto t_start = std::chrono::steady_clock::now();

        // ---- 读取header ----
        uint8_t* header = shm.getHeader();
        uint64_t fid;
        uint32_t w, h;
        std::memcpy(&fid, header, sizeof(uint64_t));
        std::memcpy(&w, header + 8, sizeof(uint32_t));
        std::memcpy(&h, header + 12, sizeof(uint32_t));

        // 帧号未变，等待新帧
        if (fid == last_fid) {
            usleep(2000);
            continue;
        }
        last_fid = fid;

        // 无效尺寸
        if (w == 0 || h == 0 || w > 4096 || h > 4096) {
            usleep(500);
            continue;
        }

        // ---- 读取图像数据 ----
        size_t data_size = static_cast<size_t>(w) * h * 3;
        if (SHM_HEADER_SIZE + data_size > cfg.shm_size) {
            std::cerr << "[警告] 图像数据超出共享内存大小" << std::endl;
            usleep(500);
            continue;
        }

        uint8_t* shm_data = shm.getData();
        cv::Mat frame(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
        std::memcpy(frame.data, shm_data, data_size);

        // ---- 垂直翻转（与Car程序一致，Python端已flip(0)）----
        cv::flip(frame, frame, 0);

        frame_counter++;

        // ---- 预处理：BGR->RGB + resize ----
        cv::Mat input;
        cv::cvtColor(frame, input, cv::COLOR_BGR2RGB);
        cv::resize(input, input, cv::Size(model_w, model_h));

        // ---- NPU推理 ----
        std::vector<cv::Mat> outputs;
        int64_t infer_us = 0;
        if (!runInference(rknn_ctx, input, outputs, &infer_us, output_attrs, n_outputs)) {
            continue;
        }
        float seg_ms = infer_us / 1000.0f;

        // ---- 后处理：argmax得到mask ----
        cv::Mat mask = postprocessSeg(outputs, w, h, model_w, model_h);

        // ---- 透视变换 → 鸟瞰图mask ----
        cv::Mat bird_mask = perspective.warp(mask);
        int bird_w = bird_mask.cols;
        int bird_h = bird_mask.rows;

        // ---- 路径规划 ----
        std::vector<cv::Point2f> centers;
        // 启用滑动平均滤波，窗口大小=5（可调整：3-7 推荐）
        // use_median=false 使用滑动平均，true 使用中值滤波
        pathplan::extractCenterline(bird_mask, centers, cfg.track_search_turn_dir);
        pathplan::smoothCenterline(centers);

        cv::Point2f aim_pt;
        bool has_aim = false;
        float heading_error = 0.0f;
        float cx = static_cast<float>(bird_w) * 0.5f;
        float cy = static_cast<float>(bird_h) * cfg.car_cy_ratio;

        if (!centers.empty()) {
            // findLookAhead 现在接收实际距离（米），内部会根据 ppm_x/ppm_y 转换
            has_aim = pathplan::findLookAhead(centers, cx, cy, cfg.aim_dist_near,
                                               cfg.ppm_x, cfg.ppm_y, aim_pt);
            if (has_aim) {
                heading_error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                                       cfg.car_length, cfg.ppm_x, cfg.ppm_y);
            }
        }

        // ---- 计算FPS ----
        auto t_now = std::chrono::steady_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_now - t_start).count();
        fps = 1000.0f / frame_ms;

        // ---- 控制台打印简要信息 ----
        std::cout << "\r[Frame " << std::setw(4) << frame_counter
                  << "] 中线:" << std::setw(3) << static_cast<int>(centers.size())
                  << " | 误差:" << std::fixed << std::setprecision(3) << heading_error
                  << " | FPS:" << std::fixed << std::setprecision(1) << fps
                  << " | Seg:" << std::fixed << std::setprecision(1) << seg_ms << "ms"
                  << std::flush;

        // ---- 可视化拼接 ----
        // 左侧: 原图 + 分割叠加
        cv::Mat display = frame.clone();
        drawSegOverlay(display, mask);

        // 右侧: 鸟瞰图可视化
        cv::Mat bird_color(bird_h, bird_w, CV_8UC3);
        bird_color.setTo(cv::Scalar(30, 30, 30));
        bird_color.setTo(cv::Scalar(0, 200, 0), bird_mask);
        if (!centers.empty()) {
            drawPathplan(bird_color, centers, aim_pt, cx, cy);
        }

        // 统一高度后再水平拼接
        int target_h = display.rows;
        if (bird_color.rows != target_h) {
            cv::resize(bird_color, bird_color, cv::Size(
                static_cast<int>(static_cast<float>(bird_color.cols) * target_h / bird_color.rows),
                target_h));
        }
        cv::Mat top;
        cv::hconcat(display, bird_color, top);
        drawHUD(top, frame_counter, static_cast<int>(centers.size()),
                heading_error, fps, seg_ms);

        // ---- HTTP推流 ----
        if (http_server && http_server->isRunning()) {
            http_server->pushFrame(top);
        }
    }

    // ===================================================
    // 7. 清理退出
    // ===================================================
    if (http_server) {
        http_server->stop();
        delete http_server;
    }
    shm.close();
    rknn_destroy(rknn_ctx);

    std::cout << "[退出] 清理完成，共处理 " << frame_counter << " 帧" << std::endl;
    return 0;
}
