//
// simble_test — 语义分割 + 目标检测 + 路径规划一体化测试程序
// 功能: 共享内存读帧 → NPU语义分割 + PP-YOLOE目标检测 → 透视变换 → 中线提取 → 纯跟踪 → HTTP推流
// 去除了: 上下位机通信、PWM控制、本地显示
// 新增: HTTP MJPEG推流，支持浏览器远程查看
//

#include "rknn_api.h"
#include "PathPlanning.h"
#include "PerspectiveMapper.h"
#include "HttpStreamServer.h"
#include "PPYoloEPostProcess.h"
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
    bool enable_segmentation = true;         // 是否启用语义分割
    // 目标检测参数（PP-YOLOE）
    std::string detection_model_path = "";
    float detection_score_thresh = 0.3f;   // 置信度阈值
    float detection_nms_thresh = 0.65f;     // NMS IoU 阈值
    bool enable_detection = false;          // 是否启用目标检测
    // 检测类别配置
    int detection_class_num = 0;
    std::vector<std::string> detection_classes;
    std::vector<std::vector<int>> detection_colors;
    // 车辆物理参数（路径规划用）
    float car_length = 0.316f;
    float car_cy_ratio = 0.95f;
    float aim_dist_near = 0.35f;
    // 标定参数
    float src_quad[8] = {0};
    int bird_width = 640;
    int bird_height = 480;
    float ppm_x = 0.0f;  // X方向像素/米
    float ppm_y = 0.0f;  // Y方向像素/米
    // 帧裁剪（flip之后、预处理之前）
    struct FrameCrop {
        bool enable = false;
        int x = 0;
        int y = 240;
        int w = 640;
        int h = 240;
    } frame_crop;
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
        if (j.contains("enable_segmentation")) cfg.enable_segmentation = j["enable_segmentation"];
        // 读取目标检测配置
        if (j.contains("detection")) {
            auto det = j["detection"];
            if (det.contains("model_path")) cfg.detection_model_path = det["model_path"];
            if (det.contains("score_thresh")) cfg.detection_score_thresh = det["score_thresh"];
            if (det.contains("nms_thresh")) cfg.detection_nms_thresh = det["nms_thresh"];
            if (det.contains("enable")) cfg.enable_detection = det["enable"];
            if (det.contains("class_num")) cfg.detection_class_num = det["class_num"];
            if (det.contains("classes")) det["classes"].get_to(cfg.detection_classes);
            if (det.contains("colors")) det["colors"].get_to(cfg.detection_colors);
        }
        if (j.contains("car_length")) cfg.car_length = j["car_length"];
        if (j.contains("car_cy_ratio")) cfg.car_cy_ratio = j["car_cy_ratio"];
        if (j.contains("aim_dist_near")) cfg.aim_dist_near = j["aim_dist_near"];
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
        // 读取帧裁剪配置
        if (j.contains("frame_crop")) {
            auto fc = j["frame_crop"];
            if (fc.contains("enable")) cfg.frame_crop.enable = fc["enable"];
            if (fc.contains("x")) cfg.frame_crop.x = fc["x"];
            if (fc.contains("y")) cfg.frame_crop.y = fc["y"];
            if (fc.contains("w")) cfg.frame_crop.w = fc["w"];
            if (fc.contains("h")) cfg.frame_crop.h = fc["h"];
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
        if (attr.n_dims == 4) {
            int channels = static_cast<int>(attr.dims[1]);
            int rows = static_cast<int>(attr.dims[2]);
            int cols = static_cast<int>(attr.dims[3]);
            outputs[i] = cv::Mat(channels, rows * cols, CV_32FC1,
                                 rk_outs[i].buf).clone();
        } else if (attr.n_dims == 3) {
            // 3D tensor: [batch, d1, d2] — 取 (d1, d2) 矩阵
            int d1 = static_cast<int>(attr.dims[1]);
            int d2 = static_cast<int>(attr.dims[2]);
            outputs[i] = cv::Mat(d1, d2, CV_32FC1,
                                 rk_outs[i].buf).clone();
        } else if (attr.n_dims == 2) {
            int rows = static_cast<int>(attr.dims[0]);
            int cols = static_cast<int>(attr.dims[1]);
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

// ============================================================================
// 分割后处理 — argmax + 贴回原图坐标
// ============================================================================
static cv::Mat postprocessSeg(const std::vector<cv::Mat>& outputs,
                              int model_w, int model_h,
                              int frame_w, int frame_h) {
    const cv::Mat& out = outputs[0];
    const float* ch0 = out.ptr<float>(0);
    const float* ch1 = out.ptr<float>(1);

    // argmax：ch0=背景, ch1=赛道
    cv::Mat seg_mask(model_h, model_w, CV_8UC1);
    const int total = model_w * model_h;
    for (int i = 0; i < total; ++i) {
        seg_mask.data[i] = (ch0[i] >= ch1[i]) ? 0 : 1;
    }

    // resize 回工作帧尺寸
    cv::Mat mask;
    cv::resize(seg_mask, mask, cv::Size(frame_w, frame_h), 0, 0, cv::INTER_NEAREST);
    return mask;
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
             float heading_error, float fps, float seg_ms,
             int n_detections = 0, float det_ms = 0.0f) {
    const int x = 12;
    int y = 24;
    int hud_h = (n_detections > 0 || det_ms > 0) ? 100 : 80;
    cv::Rect roi(x - 8, 8, 320, hud_h);
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
    y += 22;

    if (n_detections > 0 || det_ms > 0) {
        oss.str("");
        oss << std::fixed << std::setprecision(1);
        oss << "Det: " << n_detections << "个  " << det_ms << "ms";
        cv::putText(frame, oss.str(), cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, kHudText, 1);
    }
}

// 绘制目标检测结果
void drawDetections(cv::Mat& frame, const std::vector<ppyoloe::Detection>& detections,
                    int class_num, const std::vector<std::string>& class_names,
                    const std::vector<std::vector<int>>& colors) {
    if (class_num <= 0) return;
    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.x1);
        int y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2);
        int y2 = static_cast<int>(det.y2);
        int cls_id = det.class_id;
        if (cls_id < 0 || cls_id >= class_num) continue;

        cv::Scalar color(colors[cls_id][0], colors[cls_id][1], colors[cls_id][2]);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", class_names[cls_id].c_str(), det.score);
        int baseline = 0;
        cv::Size text_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5f, 1, &baseline);

        cv::rectangle(frame, cv::Point(x1, y1 - text_sz.height - 4),
                      cv::Point(x1 + text_sz.width + 4, y1), color, -1);
        cv::putText(frame, label, cv::Point(x1 + 2, y1 - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(255, 255, 255), 1);
    }
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

    // ---- 解析命令行参数 ----
    const char* config_path = "config.json";
    const char* image_path  = nullptr;
    const char* output_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" || arg == "-i") {
            if (i + 1 < argc) image_path = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg[0] != '-') {
            config_path = argv[i];  // 位置参数: 配置文件
        }
    }
    std::cout << "= simble_test: 语义分割 + 目标检测 + 路径规划一体化测试 =" << std::endl;
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
    std::cout << "语义分割: " << (cfg.enable_segmentation ? "启用" : "禁用") << std::endl;
    std::cout << "鸟瞰尺寸: " << cfg.bird_width << "x" << cfg.bird_height << std::endl;
    std::cout << "PPM_X (水平): " << cfg.ppm_x << " px/m" << std::endl;
    std::cout << "PPM_Y (垂直): " << cfg.ppm_y << " px/m" << std::endl;
    std::cout << "车参数: 长度=" << cfg.car_length << "m, 车底y比例=" << cfg.car_cy_ratio << std::endl;
    std::cout << "前视距离: " << cfg.aim_dist_near << "m" << std::endl;
    std::cout << "HTTP推流: " << (cfg.enable_http_stream ? "启用" : "禁用") << std::endl;
    if (cfg.enable_http_stream) {
        std::cout << "HTTP端口: " << cfg.http_port << std::endl;
        std::cout << "JPEG质量: " << cfg.jpeg_quality << std::endl;
    }
    if (cfg.frame_crop.enable) {
        std::cout << "帧裁剪: 启用 "
                  << "x=" << cfg.frame_crop.x << " y=" << cfg.frame_crop.y
                  << " w=" << cfg.frame_crop.w << " h=" << cfg.frame_crop.h << std::endl;
    } else {
        std::cout << "帧裁剪: 禁用" << std::endl;
    }
    std::cout << "目标检测: " << (cfg.enable_detection ? "启用" : "禁用") << std::endl;
    if (cfg.enable_detection) {
        std::cout << "检测模型: " << cfg.detection_model_path << std::endl;
        std::cout << "置信度阈值: " << cfg.detection_score_thresh << std::endl;
        std::cout << "NMS阈值: " << cfg.detection_nms_thresh << std::endl;
    }

    // ===================================================
    // 2. 初始化透视变换
    // ===================================================
    PerspectiveMapper perspective;
    perspective.init(cfg.src_quad, cfg.bird_width, cfg.bird_height);
    std::cout << "\n[透视变换] 初始化完成" << std::endl;

    // ===================================================
    // 3. 加载语义分割RKNN模型（NPU Core2）
    // ===================================================
    rknn_context rknn_ctx = 0;
    int model_w = 0, model_h = 0;
    uint32_t n_outputs = 0;
    std::vector<rknn_tensor_attr> output_attrs;
    if (cfg.enable_segmentation) {
        if (!loadRKNModel(cfg.model_path, &rknn_ctx, &model_w, &model_h, &n_outputs, &output_attrs)) {
            std::cerr << "[分割] 模型加载失败，禁用语义分割" << std::endl;
            cfg.enable_segmentation = false;
        } else {
            // 预热（NPU初始化）
            cv::Mat warm_input(model_h, model_w, CV_8UC3);
            warm_input.setTo(0);
            std::vector<cv::Mat> warm_outputs;
            int64_t warm_us = 0;
            runInference(rknn_ctx, warm_input, warm_outputs, &warm_us, output_attrs, n_outputs);
            std::cout << "[分割预热] " << (warm_us / 1000.0f) << " ms" << std::endl;
        }
    } else {
        std::cout << "[分割] 已禁用" << std::endl;
    }

    // ===================================================
    // 3b. 加载目标检测RKNN模型（NPU Core0）
    // ===================================================
    rknn_context det_ctx = 0;
    int det_model_w = 0, det_model_h = 0;
    uint32_t det_n_outputs = 0;
    std::vector<rknn_tensor_attr> det_output_attrs;
    std::vector<ppyoloe::Detection> detections;

    if (cfg.enable_detection && !cfg.detection_model_path.empty()) {
        if (!loadRKNModel(cfg.detection_model_path, &det_ctx,
                          &det_model_w, &det_model_h,
                          &det_n_outputs, &det_output_attrs)) {
            std::cerr << "[检测] 模型加载失败，禁用目标检测" << std::endl;
            cfg.enable_detection = false;
        } else {
            // 检测模型使用 Core0（与Car程序一致，seg用Core2，det用Core0）
            rknn_set_core_mask(det_ctx, RKNN_NPU_CORE_0);

            // 预热
            cv::Mat warm_det(det_model_h, det_model_w, CV_8UC3);
            warm_det.setTo(0);
            std::vector<cv::Mat> warm_det_outs;
            int64_t warm_det_us = 0;
            runInference(det_ctx, warm_det, warm_det_outs, &warm_det_us, det_output_attrs, det_n_outputs);
            std::cout << "[检测预热] " << (warm_det_us / 1000.0f) << " ms" << std::endl;
            std::cout << "[检测] 输入: " << det_model_w << "x" << det_model_h
                      << ", 输出: " << det_n_outputs << " 个tensor" << std::endl;
        }
    }

    // ===================================================
    // 4. 图片推理模式（--image）
    // ===================================================
    if (image_path) {
        std::cout << "\n= 图片推理模式 =" << std::endl;
        std::cout << "输入图片: " << image_path << std::endl;

        cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
        if (frame.empty()) {
            std::cerr << "[错误] 无法读取图片: " << image_path << std::endl;
            rknn_destroy(rknn_ctx);
            if (det_ctx != 0) rknn_destroy(det_ctx);
            return 1;
        }
        uint32_t w_img = frame.cols;
        uint32_t h_img = frame.rows;
        std::cout << "图片尺寸: " << w_img << "x" << h_img << std::endl;

        // cv::imread 是 BGR，转 RGB（与共享内存数据格式一致）
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // 垂直翻转（与流模式一致）
        cv::flip(frame, frame, 0);

        // 帧裁剪
        if (cfg.frame_crop.enable) {
            cv::Rect crop_rc(cfg.frame_crop.x, cfg.frame_crop.y,
                             cfg.frame_crop.w, cfg.frame_crop.h);
            frame = frame(crop_rc).clone();
            w_img = static_cast<uint32_t>(cfg.frame_crop.w);
            h_img = static_cast<uint32_t>(cfg.frame_crop.h);
            std::cout << "裁剪后尺寸: " << w_img << "x" << h_img << std::endl;
        }

        // ---- 语义分割 ----
        cv::Mat mask;
        if (cfg.enable_segmentation && rknn_ctx != 0) {
            cv::Mat input;
            cv::resize(frame, input, cv::Size(model_w, model_h));

            std::vector<cv::Mat> outputs;
            int64_t infer_us = 0;
            if (runInference(rknn_ctx, input, outputs, &infer_us, output_attrs, n_outputs)) {
                mask = postprocessSeg(outputs, model_w, model_h,
                                      static_cast<int>(w_img), static_cast<int>(h_img));
                std::cout << "[分割] 推理耗时: " << (infer_us / 1000.0f) << " ms" << std::endl;
            }
        }

        // ---- 目标检测 ----
        std::vector<ppyoloe::Detection> detections;
        if (cfg.enable_detection && det_ctx != 0) {
            cv::Mat det_input;
            cv::resize(frame, det_input, cv::Size(det_model_w, det_model_h));

            std::vector<cv::Mat> det_outputs;
            int64_t det_infer_us = 0;
            if (runInference(det_ctx, det_input, det_outputs, &det_infer_us,
                             det_output_attrs, det_n_outputs)) {
                if (det_outputs.size() >= 2) {
                    detections = ppyoloe::postprocess(
                        det_outputs[0].ptr<float>(0), det_outputs[1].ptr<float>(0),
                        static_cast<int>(w_img), static_cast<int>(h_img),
                        static_cast<int>(h_img), 0,
                        cfg.detection_score_thresh, cfg.detection_nms_thresh);
                }
                std::cout << "[检测] " << detections.size() << " 个目标, "
                          << (det_infer_us / 1000.0f) << " ms" << std::endl;
            }
        }

        // ---- 构建可视化 ----
        cv::Mat display = frame.clone();
        if (!mask.empty()) drawSegOverlay(display, mask);
        if (!detections.empty()) drawDetections(display, detections,
            cfg.detection_class_num, cfg.detection_classes, cfg.detection_colors);

        // 鸟瞰图
        cv::Mat bird_color;
        if (!mask.empty()) {
            cv::Mat bird_mask = perspective.warp(mask);
            bird_color = cv::Mat(bird_mask.rows, bird_mask.cols, CV_8UC3);
            bird_color.setTo(cv::Scalar(30, 30, 30));
            bird_color.setTo(cv::Scalar(0, 200, 0), bird_mask);

            std::vector<cv::Point2f> centers;
            pathplan::extractCenterline(bird_mask, centers);
            pathplan::smoothCenterline(centers);

            cv::Point2f aim_pt;
            float cx = static_cast<float>(bird_mask.cols) * 0.5f;
            float cy = static_cast<float>(bird_mask.rows) * cfg.car_cy_ratio;
            if (!centers.empty()) {
                pathplan::findLookAhead(centers, cx, cy, cfg.aim_dist_near,
                                       cfg.ppm_x, cfg.ppm_y, aim_pt);
                drawPathplan(bird_color, centers, aim_pt, cx, cy);
            }
        }

        cv::Mat top;
        if (!bird_color.empty()) {
            int target_h = display.rows;
            if (bird_color.rows != target_h) {
                cv::resize(bird_color, bird_color, cv::Size(
                    static_cast<int>(static_cast<float>(bird_color.cols) * target_h / bird_color.rows),
                    target_h));
            }
            cv::hconcat(display, bird_color, top);
        } else {
            top = display;
        }
        drawHUD(top, 0, 0, 0.0f, 0.0f, 0.0f,
                static_cast<int>(detections.size()), 0.0f);

        // ---- 保存结果 ----
        std::string out_path = output_path ? output_path : "simble_result.jpg";
        cv::Mat top_bgr;
        cv::cvtColor(top, top_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(out_path, top_bgr);
        std::cout << "结果已保存: " << out_path << std::endl;

        // 清理
        rknn_destroy(rknn_ctx);
        if (det_ctx != 0) rknn_destroy(det_ctx);
        std::cout << "[退出] 图片推理完成" << std::endl;
        return 0;
    }

    // ===================================================
    // 5. 打开共享内存
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
    // 6. 启动HTTP推流服务器
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
    // 7. 主循环
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
        // 注：frame保持RGB格式（共享内存原始格式），模型直接用，仅显示/push前转BGR

        frame_counter++;

        // ---- 帧裁剪（flip之后、预处理之前） ----
        if (cfg.frame_crop.enable) {
            cv::Rect crop(cfg.frame_crop.x, cfg.frame_crop.y,
                          cfg.frame_crop.w, cfg.frame_crop.h);
            frame = frame(crop).clone();  // clone确保内存连续
            w = cfg.frame_crop.w;
            h = cfg.frame_crop.h;
        }

        // ---- 语义分割 ----
        cv::Mat mask;
        float seg_ms = 0.0f;
        if (cfg.enable_segmentation && rknn_ctx != 0) {
            // 预处理：resize 到模型输入尺寸
            cv::Mat input;
            cv::resize(frame, input, cv::Size(model_w, model_h));

            // NPU推理
            std::vector<cv::Mat> outputs;
            int64_t infer_us = 0;
            if (!runInference(rknn_ctx, input, outputs, &infer_us, output_attrs, n_outputs)) {
                continue;
            }
            seg_ms = infer_us / 1000.0f;

            // 后处理：argmax → resize 回工作帧尺寸
            mask = postprocessSeg(outputs, model_w, model_h,
                                  static_cast<int>(w), static_cast<int>(h));
        }

        // ---- 目标检测（PP-YOLOE） ----
        float det_ms = 0.0f;
        if (cfg.enable_detection && det_ctx != 0) {
            // 预处理：resize 到检测模型输入尺寸
            cv::Mat det_input;
            cv::resize(frame, det_input, cv::Size(det_model_w, det_model_h));

            // NPU推理
            std::vector<cv::Mat> det_outputs;
            int64_t det_infer_us = 0;
            if (runInference(det_ctx, det_input, det_outputs, &det_infer_us,
                             det_output_attrs, det_n_outputs)) {
                det_ms = det_infer_us / 1000.0f;

                // 后处理：坐标映射到工作帧（全帧，无额外裁剪）
                if (det_outputs.size() >= 2) {
                    detections = ppyoloe::postprocess(
                        det_outputs[0].ptr<float>(0),  // cls_data
                        det_outputs[1].ptr<float>(0),  // reg_data
                        static_cast<int>(w), static_cast<int>(h),
                        static_cast<int>(h), 0,        // 全帧: crop_h=h, crop_top=0
                        cfg.detection_score_thresh, cfg.detection_nms_thresh);
                }
            }
        }

        // ---- 透视变换 + 路径规划（仅分割启用时） ----
        std::vector<cv::Point2f> centers;
        float heading_error = 0.0f;
        cv::Mat bird_color;
        if (cfg.enable_segmentation && !mask.empty()) {
            cv::Mat bird_mask = perspective.warp(mask);
            int bird_w = bird_mask.cols;
            int bird_h = bird_mask.rows;

            pathplan::extractCenterline(bird_mask, centers);
            pathplan::smoothCenterline(centers);

            cv::Point2f aim_pt;
            bool has_aim = false;
            float cx = static_cast<float>(bird_w) * 0.5f;
            float cy = static_cast<float>(bird_h) * cfg.car_cy_ratio;

            if (!centers.empty()) {
                has_aim = pathplan::findLookAhead(centers, cx, cy, cfg.aim_dist_near,
                                                   cfg.ppm_x, cfg.ppm_y, aim_pt);
                if (has_aim) {
                    heading_error = pathplan::headingError(cx, cy, aim_pt.x, aim_pt.y,
                                                           cfg.car_length, cfg.ppm_x, cfg.ppm_y);
                }
            }

            // 鸟瞰图可视化
            bird_color = cv::Mat(bird_h, bird_w, CV_8UC3);
            bird_color.setTo(cv::Scalar(30, 30, 30));
            bird_color.setTo(cv::Scalar(0, 200, 0), bird_mask);
            if (!centers.empty()) {
                drawPathplan(bird_color, centers, aim_pt, cx, cy);
            }
        }

        // ---- 计算FPS ----
        auto t_now = std::chrono::steady_clock::now();
        float frame_ms = std::chrono::duration<float, std::milli>(t_now - t_start).count();
        fps = 1000.0f / frame_ms;

        // ---- 控制台打印 ----
        std::cout << "\r[Frame " << std::setw(4) << frame_counter
                  << "] FPS:" << std::fixed << std::setprecision(1) << fps;
        if (cfg.enable_segmentation) {
            std::cout << " | 中线:" << std::setw(3) << static_cast<int>(centers.size())
                      << " 误差:" << std::fixed << std::setprecision(3) << heading_error
                      << " | Seg:" << std::fixed << std::setprecision(1) << seg_ms << "ms";
        }
        if (cfg.enable_detection) {
            std::cout << " Det:" << std::setw(2) << static_cast<int>(detections.size())
                      << "(" << std::fixed << std::setprecision(1) << det_ms << "ms)";
        }
        std::cout << std::flush;

        // ---- 可视化拼接 ----
        cv::Mat display = frame.clone();
        // 分割叠加（仅启用时）
        if (cfg.enable_segmentation && !mask.empty()) {
            drawSegOverlay(display, mask);
        }
        // 检测框（仅启用且有结果时）
        if (cfg.enable_detection && !detections.empty()) {
            drawDetections(display, detections,
                cfg.detection_class_num, cfg.detection_classes, cfg.detection_colors);
        }

        cv::Mat top;
        if (cfg.enable_segmentation && !bird_color.empty()) {
            // 左侧原图 + 右侧鸟瞰图，水平拼接
            int target_h = display.rows;
            if (bird_color.rows != target_h) {
                cv::resize(bird_color, bird_color, cv::Size(
                    static_cast<int>(static_cast<float>(bird_color.cols) * target_h / bird_color.rows),
                    target_h));
            }
            cv::hconcat(display, bird_color, top);
        } else {
            // 分割禁用：直接输出原图
            top = display;
        }
        drawHUD(top, frame_counter,
                static_cast<int>(centers.size()), heading_error, fps, seg_ms,
                static_cast<int>(detections.size()), det_ms);

        // ---- HTTP推流（RGB→BGR，OpenCV JPEG编码需要BGR） ----
        if (http_server && http_server->isRunning()) {
            cv::Mat top_bgr;
            cv::cvtColor(top, top_bgr, cv::COLOR_RGB2BGR);
            http_server->pushFrame(top_bgr);
        }
    }

    // ===================================================
    // 8. 清理退出
    // ===================================================
    if (http_server) {
        http_server->stop();
        delete http_server;
    }
    shm.close();
    rknn_destroy(rknn_ctx);
    if (det_ctx != 0) {
        rknn_destroy(det_ctx);
    }

    std::cout << "[退出] 清理完成，共处理 " << frame_counter << " 帧" << std::endl;
    return 0;
}
