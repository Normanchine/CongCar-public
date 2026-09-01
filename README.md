# CongCar

基于 RK3588 / Orange Pi 5B 的智能车视觉上位机。程序使用 C++17 构建多线程实时流水线：从 POSIX 共享内存读取相机帧，在 Rockchip NPU 上运行语义分割和目标检测，再完成鸟瞰变换、巡线规划、场景决策与 PD/Pure Pursuit 控制，最后通过 UART 向 STM32 底盘发送指令。

## 当前能力

- PP-LiteSeg 赛道分割与 PP-YOLOE 目标检测，使用 RKNN Runtime；
- Capture、Segmentation、Detection、Control、Display 任务并行运行；
- NormalScene 中线/边界巡线、Pure Pursuit 前视与左右边界独立 PD；
- TrafficSignScene、CarScene、PersonScene 场景处理；
- 共享结果双缓冲、性能统计和 HTTP MJPEG 推流；
- JSON 配置驱动模型、标定、控制、串口和推流参数。

## 构建与运行

```bash
git clone https://github.com/Normanchine/CongCar.git
cd CongCar
chmod +x build_all.sh run_car.sh run_calibrate.sh
./build_all.sh Release
./run_car.sh
```

启动前先运行相机共享内存服务。模型和 RKNN Runtime 未包含在仓库中，需在 RK3588 Linux 板端单独部署。

主要配置入口为 [Car/config/default.json](Car/config/default.json)，包括模型路径、透视标定、速度与左右 PD、PWM、串口、共享内存和 HTTP 推流。ai.vlm.api_key 示例为空，真实密钥只应在本地配置，切勿提交。启用推流后访问 http://<开发板IP>:5001。

## 目录

```text
Car/Car/{tasks,inference,scene,hal,common,config}/
Car/config/default.json
Car/tools/calibrate.cpp
docs/                         # 巡线、场景和参数文档
build_all.sh                 # 构建脚本
run_car.sh / run_calibrate.sh
```

## 安全与限制

这是面向真实车辆的实验性代码。首次运行请在悬空或低速条件下验证 PWM、转向方向、串口和急停链路；确认模型、标定和 NPU Runtime 后再落地运行。
