//
// simble_calib — 透视变换标定工具
// 支持两种模式：
//   1. 从共享内存读取实时帧：simble_calib [config路径]
//   2. 从图片文件读取：simble_calib <图片路径> [config路径]
//
// 流程:
// 1. 读取图像（共享内存或文件）
// 2. 鼠标依次点击 左上→右上→右下→左下 四个点
// 3. 输入鸟瞰图尺寸（默认640x480）
// 4. 输入所选区域物理尺寸 → 自动计算PPM
// 5. 预览鸟瞰效果（原图 | 鸟瞰），确认/重选
// 6. 将标定参数写入 config.json
//

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

using json = nlohmann::json;

constexpr size_t SHM_HEADER_SIZE = 16;  // fid + w + h

// ============================================================================
// 共享内存读取
// ============================================================================
struct SharedMemory {
    int fd = -1;
    void* addr = nullptr;
    size_t size = 0;

    bool open(const std::string& shm_name, size_t shm_size) {
        fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
        if (fd == -1) {
            return false;
        }
        addr = mmap(nullptr, shm_size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            return false;
        }
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

    bool isValid() const { return addr != nullptr && addr != MAP_FAILED; }
};

// 从共享内存读取一帧
static bool readFrameFromShm(const std::string& shm_name, size_t shm_size,
                              cv::Mat& out_frame) {
    SharedMemory shm;
    if (!shm.open(shm_name, shm_size)) {
        std::cerr << "[错误] 无法打开共享内存: " << shm_name << std::endl;
        std::cerr << "[提示] 请确保 Python 视频推流服务已启动" << std::endl;
        return false;
    }

    // 等待有效帧
    uint64_t last_fid = 0;
    int wait_count = 0;
    while (wait_count < 100) {  // 最多等待1秒
        uint8_t* header = static_cast<uint8_t*>(shm.addr);
        uint64_t fid;
        uint32_t w, h;
        std::memcpy(&fid, header, sizeof(uint64_t));
        std::memcpy(&w, header + 8, sizeof(uint32_t));
        std::memcpy(&h, header + 12, sizeof(uint32_t));

        if (fid != last_fid && w > 0 && h > 0 && w < 4096 && h < 4096) {
            // 有效帧
            size_t data_size = static_cast<size_t>(w) * h * 3;
            if (SHM_HEADER_SIZE + data_size <= shm_size) {
                cv::Mat frame(static_cast<int>(h), static_cast<int>(w), CV_8UC3);
                std::memcpy(frame.data,
                           static_cast<uint8_t*>(shm.addr) + SHM_HEADER_SIZE,
                           data_size);
                cv::flip(frame, frame, 0);  // 垂直翻转
                out_frame = frame;
                shm.close();
                return true;
            }
        }
        last_fid = fid;
        usleep(10000);  // 10ms
        wait_count++;
    }

    std::cerr << "[错误] 共享内存无有效帧" << std::endl;
    shm.close();
    return false;
}

// ============================================================================
// 标定状态和交互
// ============================================================================
struct CalibState {
    cv::Mat image;
    cv::Mat display;
    std::vector<cv::Point2f> src_pts;
};

static void onMouse(int event, int x, int y, int flags, void* userdata) {
    auto* state = static_cast<CalibState*>(userdata);
    if (event != cv::EVENT_LBUTTONDOWN) return;
    if (state->src_pts.size() >= 4) return;

    state->src_pts.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
    std::cout << "  点 " << state->src_pts.size() << ": ("
              << x << ", " << y << ")" << std::endl;

    cv::circle(state->display, cv::Point(x, y), 6, cv::Scalar(0, 0, 255), -1);

    if (state->src_pts.size() >= 2) {
        std::vector<cv::Point> pts;
        for (const auto& p : state->src_pts) {
            pts.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        }
        std::vector<std::vector<cv::Point>> contours = { pts };
        cv::polylines(state->display, contours, (state->src_pts.size() == 4),
                      cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("选择4个角点 (左上→右上→右下→左下)", state->display);
}

static bool pickPoints(const cv::Mat& image, std::vector<cv::Point2f>& out_pts) {
    CalibState state;
    state.image = image.clone();
    state.display = image.clone();

    cv::namedWindow("选择4个角点 (左上→右上→右下→左下)");
    cv::setMouseCallback("选择4个角点 (左上→右上→右下→左下)", onMouse, &state);

    std::cout << "\n请在图片上依次点击: 左上 → 右上 → 右下 → 左下" << std::endl;
    std::cout << "按 ESC 重新选择, 选完后按任意键继续..." << std::endl;

    cv::imshow("选择4个角点 (左上→右上→右下→左下)", state.display);

    while (state.src_pts.size() < 4) {
        int key = cv::waitKey(10) & 0xFF;
        if (key == 27) {
            state.src_pts.clear();
            state.display = state.image.clone();
            cv::imshow("选择4个角点 (左上→右上→右下→左下)", state.display);
            std::cout << "\n已重置，重新选择" << std::endl;
        }
    }

    cv::waitKey(0);
    cv::destroyWindow("选择4个角点 (左上→右上→右下→左下)");

    out_pts = state.src_pts;
    return true;
}

static bool getBirdParams(int img_w, int img_h,
                           int& bird_w, int& bird_h, float& ppm_x, float& ppm_y) {
    std::cout << "\n图片原始尺寸: " << img_w << " x " << img_h << std::endl;

    std::cout << "鸟瞰图宽度 (推荐 640): ";
    std::string buf_w;
    std::getline(std::cin, buf_w);
    bird_w = buf_w.empty() ? 640 : std::stoi(buf_w);

    std::cout << "鸟瞰图高度 (推荐 480): ";
    std::string buf_h;
    std::getline(std::cin, buf_h);
    bird_h = buf_h.empty() ? 480 : std::stoi(buf_h);

    std::cout << "所选区域实际宽度 (米, 如 1.0): ";
    std::string buf_rw;
    std::getline(std::cin, buf_rw);
    float rw = buf_rw.empty() ? 1.0f : std::stof(buf_rw);

    std::cout << "所选区域实际长度 (米, 如 2.0): ";
    std::string buf_rh;
    std::getline(std::cin, buf_rh);
    float rh = buf_rh.empty() ? 2.0f : std::stof(buf_rh);

    // 分别计算 X 和 Y 方向的 PPM
    ppm_x = std::round(bird_w / rw * 100.0f) / 100.0f;  // 水平方向
    ppm_y = std::round(bird_h / rh * 100.0f) / 100.0f;  // 垂直方向
    return true;
}

static bool preview(const cv::Mat& image,
                    const std::vector<cv::Point2f>& src_pts,
                    int bird_w, int bird_h, float ppm_x, float ppm_y) {
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(bird_w - 1), 0),
        cv::Point2f(static_cast<float>(bird_w - 1), static_cast<float>(bird_h - 1)),
        cv::Point2f(0, static_cast<float>(bird_h - 1))
    };
    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);

    cv::Mat bird_view;
    cv::warpPerspective(image, bird_view, M, cv::Size(bird_w, bird_h));

    std::cout << "\nPPM_X (水平) = " << ppm_x << " px/m" << std::endl;
    std::cout << "PPM_Y (垂直) = " << ppm_y << " px/m" << std::endl;

    cv::Mat src_scaled;
    cv::resize(image, src_scaled, cv::Size(bird_w, bird_h));

    cv::Mat preview_img;
    cv::hconcat(src_scaled, bird_view, preview_img);

    cv::imshow("原图(缩放) | 鸟瞰图  — 任意键确认, ESC重新标定", preview_img);

    int key = cv::waitKey(0) & 0xFF;
    cv::destroyWindow("原图(缩放) | 鸟瞰图  — 任意键确认, ESC重新标定");

    return (key != 27);
}

static void saveConfig(const std::string& config_path,
                       const std::vector<cv::Point2f>& src_pts,
                       int bird_w, int bird_h, float ppm_x, float ppm_y) {
    if (config_path.empty()) {
        std::cout << "\n未指定配置文件，跳过写入" << std::endl;
        std::cout << "\n================== 标定结果（手动填写） ==================" << std::endl;
        std::cout << "\"calib\": {" << std::endl;
        std::cout << "  \"src_quad\": [";
        for (size_t i = 0; i < src_pts.size(); ++i) {
            std::cout << src_pts[i].x << ", " << src_pts[i].y;
            if (i < src_pts.size() - 1) std::cout << ", ";
        }
        std::cout << "]," << std::endl;
        std::cout << "  \"bird_width\": " << bird_w << "," << std::endl;
        std::cout << "  \"bird_height\": " << bird_h << "," << std::endl;
        std::cout << "  \"ppm_x\": " << ppm_x << "," << std::endl;
        std::cout << "  \"ppm_y\": " << ppm_y << std::endl;
        std::cout << "}" << std::endl;
        return;
    }

    json config;
    std::ifstream in(config_path);
    if (in.is_open()) {
        try {
            config = json::parse(in);
        } catch (...) {
            config = json::object();
        }
        in.close();
    }

    json calib;
    json src_quad = json::array();
    for (const auto& pt : src_pts) {
        src_quad.push_back(pt.x);
        src_quad.push_back(pt.y);
    }
    calib["src_quad"] = src_quad;
    calib["bird_width"] = bird_w;
    calib["bird_height"] = bird_h;
    calib["ppm_x"] = ppm_x;
    calib["ppm_y"] = ppm_y;
    config["calib"] = calib;

    // 保留共享内存配置
    if (!config.contains("shm_name")) {
        config["shm_name"] = "shm_ar_video";
    }
    if (!config.contains("shm_size")) {
        config["shm_size"] = 10485760;
    }

    std::ofstream out(config_path);
    out << config.dump(2) << std::endl;
    out.close();

    std::cout << "\n标定参数已写入: " << config_path << std::endl;
}

// ============================================================================
// 主入口
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "= simble_calib: 透视变换标定工具 =" << std::endl;

    cv::Mat image;
    std::string config_path;
    std::string shm_name = "shm_ar_video";
    size_t shm_size = 10 * 1024 * 1024;
    bool crop_enabled = false;
    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;

    // 解析 --crop 参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--crop" && i + 1 < argc) {
            std::string crop_str = argv[++i];
            sscanf(crop_str.c_str(), "%d,%d,%d,%d", &crop_x, &crop_y, &crop_w, &crop_h);
            crop_enabled = true;
        }
    }
    // 过滤掉 --crop，提取位置参数
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--crop") { ++i; continue; }
        positional.push_back(arg);
    }

    // 解析参数
    if (positional.empty()) {
        // 无参数：从共享内存读取
        std::cout << "\n[模式] 从共享内存读取实时帧" << std::endl;
        std::cout << "共享内存: " << shm_name << std::endl;
        std::cout << "提示: 请确保 Python 视频推流服务已启动" << std::endl;
        std::cout << "等待帧..." << std::endl;

        if (!readFrameFromShm(shm_name, shm_size, image)) {
            return 1;
        }
        std::cout << "已获取帧: " << image.cols << " x " << image.rows << std::endl;

    } else if (positional.size() == 1) {
        // 一个参数：可能是图片路径或config路径
        std::string arg1 = positional[0];

        // 判断是图片还是json
        if (arg1.find(".json") != std::string::npos ||
            arg1.find(".JSON") != std::string::npos) {
            // 是 config 路径，从共享内存读取
            config_path = arg1;
            std::cout << "\n[模式] 从共享内存读取实时帧" << std::endl;
            std::cout << "配置输出: " << config_path << std::endl;

            // 尝试从配置读取共享内存参数
            std::ifstream in(config_path);
            if (in.is_open()) {
                try {
                    json cfg = json::parse(in);
                    if (cfg.contains("shm_name")) shm_name = cfg["shm_name"];
                    if (cfg.contains("shm_size")) shm_size = cfg["shm_size"];
                } catch (...) {}
                in.close();
            }

            std::cout << "共享内存: " << shm_name << std::endl;
            std::cout << "等待帧..." << std::endl;

            if (!readFrameFromShm(shm_name, shm_size, image)) {
                return 1;
            }
            std::cout << "已获取帧: " << image.cols << " x " << image.rows << std::endl;

        } else {
            // 是图片路径
            std::cout << "\n[模式] 从图片文件读取" << std::endl;
            std::cout << "图片: " << arg1 << std::endl;

            image = cv::imread(arg1);
            if (image.empty()) {
                std::cerr << "[错误] 无法读取图片: " << arg1 << std::endl;
                return 1;
            }
        }

    } else {
        // 两个参数：图片路径 + config路径
        std::string image_path = positional[0];
        config_path = positional[1];

        std::cout << "\n[模式] 从图片文件读取" << std::endl;
        std::cout << "图片: " << image_path << std::endl;
        std::cout << "配置输出: " << config_path << std::endl;

        image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "[错误] 无法读取图片: " << image_path << std::endl;
            return 1;
        }
    }

    int img_w = image.cols;
    int img_h = image.rows;
    std::cout << "图片尺寸: " << img_w << " x " << img_h << std::endl;

    // 裁剪图像（标定在工作帧上做，坐标天然匹配）
    if (crop_enabled) {
        cv::Rect crop_rc(crop_x, crop_y, crop_w, crop_h);
        image = image(crop_rc).clone();
        img_w = image.cols;
        img_h = image.rows;
        std::cout << "裁剪至: " << img_w << " x " << img_h
                  << " (crop " << crop_x << "," << crop_y
                  << " " << crop_w << "x" << crop_h << ")" << std::endl;
    }

    std::vector<cv::Point2f> src_pts;
    int bird_w = 640, bird_h = 480;
    float ppm_x = 0.0f, ppm_y = 0.0f;

    while (true) {
        if (!pickPoints(image, src_pts)) {
            return 1;
        }

        if (!getBirdParams(img_w, img_h, bird_w, bird_h, ppm_x, ppm_y)) {
            return 1;
        }

        if (preview(image, src_pts, bird_w, bird_h, ppm_x, ppm_y)) {
            break;
        }
    }

    saveConfig(config_path, src_pts, bird_w, bird_h, ppm_x, ppm_y);

    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(bird_w - 1), 0),
        cv::Point2f(static_cast<float>(bird_w - 1), static_cast<float>(bird_h - 1)),
        cv::Point2f(0, static_cast<float>(bird_h - 1))
    };
    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    std::cout << "\n变换矩阵:" << std::endl;
    std::cout << M << std::endl;

    return 0;
}
