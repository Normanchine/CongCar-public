# CongCar

CongCar 是运行在 RK3588 / Orange Pi 5B 上的智能车视觉上位机。它以 C++17 实现实时多线程流水线，从摄像头共享内存读取图像，在 Rockchip NPU 上执行赛道分割和目标检测，经过鸟瞰变换、场景约束与路径控制后，通过 UART 向 STM32 底盘发送速度和转向指令。

项目定位是“板端控制程序与实验记录”，不包含训练数据、模型权重、下位机固件或特定车辆的完整接线方案。

> 🏆 团队在第 21 届全国大学生智能汽车竞赛中获得**国家二等奖**。本页与[项目经历页](https://zhangyicong.xyz/zh/projects/智能车从能跑到稳定高速/)（[EN](https://zhangyicong.xyz/en/projects/智能车从能跑到稳定高速/) · [日本語](https://zhangyicong.xyz/ja/projects/智能车从能跑到稳定高速/)）说明个人参与范围与工程判断；成绩为团队协作成果。

速度能力是逐级调出来的：`工作日志/` 保存了从 `0.2 m/s` 到 `1.5 m/s（稳定版）` 各阶段的配置快照，对应“先能跑、再跑快、后跑稳”的验证顺序。

## 系统架构

```text
Python 摄像头服务
        │ POSIX shared memory
        ▼
CaptureTask ──┬── SegmentationTask ──→ 赛道掩码
              ├── DetectionTask ─────→ sign / person / car
              └── DisplayTask ───────→ HUD + HTTP MJPEG
                                      │
                                      ▼
                          ControlTask / 场景约束层
                          透视变换 → 中线/边界 → 路径跟踪
                                      │
                              Pure Pursuit + PD
                                      │
                          UART (/dev/ttyUSB0) → STM32
```

任务之间通过有界帧队列和共享结果缓冲交换数据。队列拥塞时优先丢弃旧帧，以降低控制链路延迟；推理线程记录耗时，显示和 HTTP 推流使用最近可用结果。

## 当前实现

### 视觉与推理

- PP-LiteSeg 语义分割提取赛道区域；
- PP-YOLOE 目标检测识别交通标志、行人和车辆；
- 使用 RKNN Runtime 加载 `.rknn` 模型，可按配置绑定 RK3588 NPU 核心；
- 支持输入尺寸、阈值、NMS 和类别映射配置；
- VLM 仅用于交通标志场景的语义补充，API Key 不随仓库提供。

### 巡线与控制

- 透视标定将相机图像转换为鸟瞰坐标；
- NormalScene 支持中线或指定边界巡线；
- Pure Pursuit 计算前视目标，PD 控制输出转向 PWM；
- 左右边界可使用独立的 P/D 参数；
- 支持丢线后的渐进式降速、转向保持与恢复；
- 控制参数集中在 `control` 配置段，场景只覆盖必要的行为差异。

### 场景处理

当前代码包含以下场景：

| 场景 | 作用 |
| --- | --- |
| `NormalScene` | 常规中线/边界巡线 |
| `TrafficSignScene` | 交通标志判别、停车/倒车、右转锁定与短时盲开 |
| `CarScene` | 车辆障碍相关的避让约束 |
| `PersonScene` | 行人检测后的减速或停车约束 |

场景选择由检测结果和控制上下文驱动。交通标志的 VLM 结果属于外部服务输入，网络超时、无效 JSON 或密钥缺失时应按日志排查，不应直接视为车辆安全策略。

### 通信与可视化

- UART 向 STM32 发送速度和 PWM，并接收底盘状态；
- HTTP MJPEG 服务默认端口为 5001，可在浏览器观察原图、分割、检测、鸟瞰和 HUD；
- 日志包含任务启动、模型加载、推理耗时、串口和控制状态；
- 结果缓冲避免显示线程阻塞控制线程。

## 硬件与软件要求

| 项目 | 要求 |
| --- | --- |
| 主控 | RK3588，示例板为 Orange Pi 5B |
| 系统 | Linux ARM64 |
| 编译器 | GNU C++17、CMake 3.16+、make |
| 图像库 | OpenCV 4.x |
| JSON | nlohmann/json |
| 推理运行时 | Rockchip `librknnrt.so` |
| 输入 | 写入 POSIX 共享内存的摄像头服务 |
| 底盘 | STM32 控制板及可用 UART |
| 模型 | PP-LiteSeg / PP-YOLOE 的 `.rknn` 文件 |

模型、RKNN Runtime 和摄像头服务不在本仓库中。请根据板端发行版安装 OpenCV、CMake 和 nlohmann/json，并把 Runtime 与模型部署到设备上。

## 获取、构建和运行

```bash
git clone https://github.com/Normanchine/CongCar-public.git
cd CongCar

chmod +x build_all.sh run_car.sh run_calibrate.sh
./build_all.sh Release

# 默认使用 Car/config/default.json
./run_car.sh

# 指定配置
./run_car.sh Car/config/default.json

# 运行透视标定工具
./run_calibrate.sh
```

构建脚本当前生成：

```text
Car/build/car        # 主程序
Car/build/car_calib  # 透视标定工具
```

如果只想手动构建：

```bash
cd Car
mkdir -p build
cd build
cmake ../Car -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 配置说明

运行配置入口为 [Car/config/default.json](Car/config/default.json)。

| 配置段 | 关键字段 | 说明 |
| --- | --- | --- |
| 顶层 | `shm_name`, `shm_size` | 与摄像头服务匹配的共享内存 |
| `control` | `max_speed`, `pid_p`, `pid_d` | 速度、左侧/默认边界控制 |
| `control` | `pid_p_right`, `pid_d_right` | 右侧边界控制 |
| `control` | `pwm_mid`, `pwm_min`, `pwm_max` | 转向 PWM 范围和中位值 |
| `ai.segmentation` | `model_path`, `input_w`, `input_h` | 分割模型与输入尺寸 |
| `ai.detection` | `model_path`, `confidence_threshold` | 检测模型与阈值 |
| `ai.vlm` | `base_url`, `model`, `api_key` | 可选 VLM 服务配置 |
| `calib` | `src_quad`, `bird_width`, `bird_height` | 透视变换标定 |
| `uart` | `port`, `baud` | STM32 串口设备 |
| `frame_crop` | `x`, `y`, `w`, `h` | 输入图像裁切 |
| `http_stream` | `port`, `jpeg_quality`, `enable` | 远程调试推流 |

VLM 密钥示例保持为空。请在部署设备的本地配置文件中填写，不要提交真实密钥，也不要把包含密钥的配置复制到公共 issue 或日志中。

## 推荐调试顺序

1. 确认摄像头服务正在运行，且共享内存名称、尺寸和帧格式一致；
2. 单独验证 OpenCV 能打开摄像头或读取共享内存；
3. 检查 `.rknn` 路径、输入尺寸和 `librknnrt.so`；
4. 使用 `run_calibrate.sh` 完成鸟瞰透视四点；
5. 车辆悬空时确认 UART、PWM 中位值和转向正负方向；
6. 打开 HTTP 推流，观察分割区域、检测框、中线和控制 HUD；
7. 低速运行后再逐步调节 `max_speed`、前视距离、左右 PD 和丢线参数；
8. 最后才启用交通标志 VLM 和复杂场景约束。

## 常见问题

### 找不到模型或 Runtime

检查配置中的模型绝对路径，并确认设备上存在 `librknnrt.so`。模型输入尺寸必须与配置及导出模型一致，量化模型还要确认输入数据类型。

### 没有图像

检查摄像头服务是否先启动、共享内存名称是否为 `shm_ar_video`，以及 `frame_crop` 是否超出实际帧尺寸。

### 推流无法访问

确认 `http_stream.enable=true`、端口未被占用，并从同一局域网访问 `http://<开发板IP>:5001`。通过 SSH 访问时可能还需要端口转发。

### 车辆方向相反或 PWM 异常

先不要让车辆落地运行。检查 `pwm_mid`、`steering_invert`、串口设备名和 STM32 协议，再在悬空或支架上验证。

## 项目结构

```text
Car/
├─ Car/main.cpp                 # 程序入口与信号处理
├─ Car/tasks/                   # Capture、推理、控制、显示任务
├─ Car/inference/               # RKNN 推理与后处理
├─ Car/scene/                   # 场景与路径控制
├─ Car/hal/                     # 共享内存、UART、RKNN 接口
├─ Car/common/                  # 队列、结果、日志、HTTP 推流
├─ Car/config/                  # 配置结构与解析
└─ tools/calibrate.cpp          # 透视标定工具
docs/                           # 巡线、场景、参数和架构文档
build_all.sh                    # 构建脚本
run_car.sh                      # 主程序启动脚本
run_calibrate.sh                # 标定启动脚本
```

更多实现细节见 [docs/](docs/) 和仓库内的技术文档。

## 安全边界

这是面向真实车辆的实验性代码，不是经过认证的自动驾驶系统。首次运行必须：

- 在悬空、支架或可立即断电的环境验证 PWM 和转向；
- 配置并测试独立急停链路；
- 限制最高速度，确认丢线和通信异常时的行为；
- 不把盲开、VLM 决策或网络连接当作安全保护；
- 不在公开仓库、日志或 issue 中发布 API Key、局域网凭据和个人设备信息。

## 贡献

欢迎提交问题和改进建议。涉及控制参数、场景策略或硬件协议的改动，请同时说明测试环境、输入数据和已知限制；不要把未验证的实车行为描述为稳定功能。
