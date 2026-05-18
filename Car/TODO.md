# 待完成工作

---

## 1. 5个具体线程任务类

架构图中每个线程对应一个 class，继承 IThreadTask，实现 run()。

| 线程 | 架构节点 | run() 需要做的事 |
|------|---------|-----------------|
| **CaptureTask** | A（采集线程） | ✅ 已完成：共享内存读取 → 深拷贝 → 入队 Q2/Q3，含断线重连 + 动态睡眠 + PerfStats 计时 |
| **SegmentationTask** | B（语义分割） | ⬜ 已移除（架构保留，后续重写） |
| **DetectionTask** | C（目标检测+OCR） | ✅ 已完成：Q3→预处理→RKNN推理→DFL解码+NMS→发布det_result(双缓冲)→选最底目标→SceneManager，调试模式推原帧到显示队列 |
| **ControlTask** | CT（控制线程） | ✅ 已完成：UartIO open/startReceive → 等待按键启动 → 循环[SceneManager::run(ctx)→sendControl→PerfStats]，按键急停/恢复 |
| **DisplayTask** | D（显示线程） | ✅ 已完成：从 Q1 tryConsume → 读 SharedResult(seg+det) → 绘制覆盖+检测框 → imshow + waitKey，ESC 键退出，含 PerfStats 计时 |

每个类用 `REGISTER_THREAD_TASK(ClassName)` 注册。

---

## 2. 线程池任务

| 任务 | 提交者 | 说明 |
|------|--------|------|
| T1 性能统计 | 定时器（Car内部，每5秒） | ✅ 已完成：定时器直接调用 PerfStats::report() |
| 线程池 | — | 保留 | 留作后续调试模块扩展 |
| | |

---

## 3. 具体场景类

实现 IScene 子类，用 `REGISTER_SCENE(ClassName)` 注册。

| 场景 | 状态 |
|------|------|
| NormalScene | ✅ 已完成：mask列扫描中线 → 纯跟踪前视点 → 航向误差 → PD → 输出 steering_pwm / target_speed |
| 直道 / 弯道 / 十字路口 / 障碍物 | ⬜ 待实现 |

---

## 4. JSON 配置加载

`Car::init()` 中 TODO 项：命令行参数解析 + JSON 反序列化 → 填充 AppConfig。

需引入 JSON 库（如 nlohmann/json）。

---

## 5. 下位机通信协议

✅ 已完成：UartIO HAL（hal/uart/UartIO.h/.cpp）+ ControlTask 持有调用，协议兼容 sasu_icar2025，支持按键启动/急停。

---

## 6. 摄像头 / NPU 推理接口

- 采集线程：✅ 通过共享内存对接 Python 服务端（非直接摄像头驱动）
- NPU 推理：✅ RKNNEngine（dlopen librknnrt.so）+ rknn_runtime_dl.h，Seg/Det 均已接入
- OCR 检测：待后续补充
