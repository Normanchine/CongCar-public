# CongCar

基于 **RK3588 / Orange Pi 5B** 的智能车视觉上位机。项目以 C++17 构建多线程实时流水线：从共享内存获取相机帧，在 Rockchip NPU 上完成语义分割与目标检测，再进行巡线规划与 PD 控制，最终通过 UART 向 STM32 底盘发送控制指令。

> 项目面向 Linux ARM 板端部署。运行真实车辆前，请先完成模型路径、透视标定、串口和 PWM 参数校验，并确保车辆具备可靠急停措施。

## 系统闭环

```text
Python 摄像头服务
        │ POSIX Shared Memory
        ▼
CaptureTask ──┬──→ SegmentationTask ──→ 赛道掩码
              ├──→ DetectionTask ─────→ 目标检测结果
              └──→ DisplayTask ───────→ HTTP MJPEG 推流
                                      │
                                      ▼
                              ControlTask / SceneManager
                              透视变换 → 中线提取 → Pure Pursuit + PD
                                      │
                                      ▼
                             UART (/dev/ttyUSB0) → STM32 底盘
```

## 特性

- **RKNN NPU 推理**：PP-LiteSeg 语义分割与 YOLOv8 目标检测，分别绑定 RK3588 的独立 NPU 核心。
- **低延迟多线程流水线**：采集、分割、检测、控制、显示分离；帧队列满时丢弃旧帧，优先保持实时性。
- **无锁共享结果**：`atomic<shared_ptr>` 双缓冲将 AI 结果提供给控制与显示任务。
- **场景化控制**：`SceneManager` 管理巡线等场景，当前 `NormalScene` 使用鸟瞰中线、Pure Pursuit 与 PD 控制转向。
- **串口控制协议**：向 STM32 下发速度和 PWM，并接收按键/急停等状态。
- **远程可视化**：内置 HTTP MJPEG 推流，浏览器即可查看分割、检测、HUD 与鸟瞰视图。
- **JSON 配置驱动**：模型路径、阈值、标定参数、控制参数、串口与推流设置均集中在配置文件中。

## 硬件与软件环境

| 项目 | 说明 |
|---|---|
| 主控 | RK3588 / Orange Pi 5B |
| NPU | Rockchip RKNN Runtime (`librknnrt.so`) |
| 摄像头输入 | Python 服务写入 POSIX 共享内存 |
| 下位机 | STM32 底盘控制板，经 UART 通信 |
| 语言 | C++17 |
| 依赖 | CMake ≥ 3.16、OpenCV 4.x、nlohmann/json、RKNN Runtime |

## 快速开始

### 1. 获取依赖

在 Orange Pi / RK3588 Linux 环境中准备 OpenCV、RKNN Runtime 与 JSON 库：

```bash
sudo apt update
sudo apt install -y build-essential cmake nlohmann-json3-dev
# OpenCV 4.x 和 librknnrt.so 需按板端环境安装或部署
```

### 2. 配置项目

编辑 [`Car/config/default.json`](Car/config/default.json)：

- `ai.*_model_path`：填写板端 `.rknn` 模型绝对路径；
- `uart.port` / `uart.baud`：确认 STM32 串口设备与波特率；
- `calib`：完成透视标定后写入四点与鸟瞰参数；
- `control`：按车辆实际尺寸、PWM 范围和赛道调节控制参数；
- `http_stream`：设置 HTTP 推流端口和 JPEG 质量。

### 3. 构建

```bash
git clone https://github.com/Normanchine/CongCar.git
cd CongCar
chmod +x build_all.sh run_*.sh

# 构建主程序、分割测试和模拟/标定工具
./build_all.sh Release
```

构建后的主要可执行文件：

```text
Car/build/car              # 智能车主程序
Car/build/car_calib        # 主程序透视标定工具
seg/build/seg_test         # 语义分割测试
simble/build/simble_test   # 巡线一体化测试
simble/build/simble_calib  # 独立透视标定工具
```

### 4. 运行

先启动负责写入共享内存的摄像头服务，再启动上位机：

```bash
./run_car.sh
# 或指定配置文件
./run_car.sh Car/config/default.json
```

如果配置中启用了 HTTP 推流，浏览器访问：

```text
http://<开发板IP>:5001
```

## 项目结构

```text
.
├── Car/
│   ├── Car/
│   │   ├── main.cpp              # 程序入口与信号处理
│   │   ├── tasks/                # 采集、分割、检测、控制、显示任务
│   │   ├── thread/               # 生命周期、任务工厂、共享上下文
│   │   ├── inference/            # RKNN 推理与 YOLO 后处理
│   │   ├── scene/                # 场景、路径规划与控制算法
│   │   ├── hal/                  # Shared Memory、UART、RKNN 接口
│   │   ├── common/               # 队列、结果双缓冲、性能统计、推流
│   │   └── config/               # C++ 配置结构与解析
│   ├── config/default.json       # 默认运行配置
│   └── tools/calibrate.cpp       # 标定工具
├── seg/                          # 语义分割独立测试
├── simble/                       # 巡线/标定独立测试
├── docs/                         # 参数测量与标定使用说明
├── build_all.sh                  # 一键构建全部目标
├── run_car.sh                    # 启动主程序
└── 技术文档.md                    # 详细架构与实现说明
```

## 运行参数与调试

`Car/config/default.json` 是唯一运行配置入口。建议按以下顺序调试：

1. 验证共享内存名称和相机图像尺寸；
2. 检查 `.rknn` 模型路径及 NPU Runtime；
3. 使用标定工具完成鸟瞰透视变换；
4. 在低速、悬空或安全支撑条件下校验 UART 与 PWM 中位值；
5. 通过 HTTP 推流观察分割、检测、中线和控制状态；
6. 最后逐步调节 `pid_p`、`pid_d`、前视距离和速度上限。

更详细的线程模型、NPU 核心分配、串口协议和扩展方式见 [技术文档.md](技术文档.md)。

## 安全提示

- 不要在未确认 `pwm_mid`、转向方向和急停链路前让车辆落地运行。
- 修改模型输入尺寸或量化模型时，需同步检查 RKNN 输入数据类型；量化模型通常使用 `UINT8`。
- 串口设备名可能随系统重启变化，建议通过稳定的 udev 规则或启动前检查确认设备。
