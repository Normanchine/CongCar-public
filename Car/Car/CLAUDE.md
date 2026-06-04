# CLAUDE.md

C++17 智能车上位机，运行于 RK3588 (Orange Pi 5B)。

## 构建

```bash
cd Car && mkdir -p build && cd build
cmake ../Car -DCMAKE_BUILD_TYPE=Release
make -j8
./car [config/default.json]
```

依赖: C++17, OpenCV 4.x, librknnrt.so (RK3588 NPU SDK), nlohmann-json3-dev。

## 架构

所有工作在线程任务中执行，继承 `IThreadTask`（只有 `run()` 纯虚函数），通过 `REGISTER_THREAD_TASK` 宏注册。共享状态集中在 `SharedContext`。

**生产线程**: CaptureTask(共享内存采集+预处理) → SegmentationTask(NPU语义分割) → DetectionTask(NPU YOLO检测) → ControlTask(场景执行+串口下发)
**仅调试**: DisplayTask(取显示帧 → 画分割覆盖+检测框 → imshow)

通信: `FrameQueue<T>` 环形缓冲传帧；`SharedResult<T>` atomic shared_ptr 双缓冲传AI结果。
SegTask 产出新分割结果后通过 `condition_variable` 唤醒 ControlTask。

## 数据流

```
CaptureTask:
  ├─ segmentation_queue → UINT8, BGR→RGB + resize(160x160)     [量化模型直传]
  ├─ detection_queue    → UINT8, BGR→RGB + resize(640x640)     [同上]
  └─ display_queue      → 原始 BGR 帧                          [debug]

SegmentationTask:
  seg_queue → NPU(Core2) → argmax → resize原图 → seg_result.publish() → seg_cv_.notify_one()

DetectionTask:
  det_queue → NPU(Core0) → YOLO decode+NMS → det_result.publish() + scene切换(仅变化时)

DisplayTask:
  display_queue(原始帧) + seg + det → 叠加 + HUD(场景/PWM/速度/FPS) + PIP(鸟瞰图中线/前视点) → imshow

ControlTask:
  cv.wait_for(50ms) ← 被 SegTask notify 唤醒，或超时兜底
  scene_manager.run(ctx) → steering_pwm/target_speed → UART sendControl

退出: Ctrl+C / 赛道丢失 → stop=true → 各线程安全退出 → sendControl(0,neutral)
```

## 线程启动顺序

```cpp
launchTask("CaptureTask");
launchTask("SegmentationTask");    // 在 Detection 之前
launchTask("DetectionTask");
launchTask("ControlTask");
if (debug) launchTask("DisplayTask");
```

## NPU 核心分配

每个任务独立实例化 `RKNNEngine`，通过 `load(path, core_id)` 绑定核心：

| 任务 | NPU 核心 | 引擎初始化 |
|------|----------|-----------|
| DetectionTask | Core 0 | `engine.load(path, 0)` |
| SegmentationTask | Core 2 | `engine.load(path, 2)` |
| OCR (预留) | Core 1 | — |

## PerfStats

5 个槽位: Capture / Seg / Detect / Control / Display。AI 任务通过 `recordStages(slot, total_us, infer_us, post_us)` 记录墙钟耗时 + 推理 + 后处理。T1 定时器每 5 秒调用 `report()`。

## 强制规则

- 新任务 .cpp: `REGISTER_THREAD_TASK` 前必须 `#include "../thread/ThreadTaskFactory.h"` + `<memory>`，宏和构造函数之间不能有任何代码
- 新任务 .cpp 必须添加到 `CMakeLists.txt` 的 `add_executable` 中，否则静态注册不触发 → "[Car] 未注册的线程任务"
- `hal/rknn/rknn_api.h` — 官方 SDK 头文件，禁止修改（ABI 兼容）
- 新场景: 继承 `IScene`，`REGISTER_SCENE(Name)`，由 ControlTask 调用 `SceneManager::run(ctx)`
- 共享内存帧格式: 16字节头(fid+w+h) + 裸RGB数据，CaptureTask 不做翻转/色域转换
- **RKNN 模型类型匹配**: 量化模型 → `RKNN_TENSOR_UINT8`，浮点模型 → `RKNN_TENSOR_FLOAT32`。类型不匹配直接段错误
- **模型尺寸从 config/engine 动态读取**，禁止硬编码。`engine.inputWidth/Height()` 或 `config.ai.segmentation/detection_input_w/h`
- 预处理在 CaptureTask 中进行，推理任务不做预处理

## 独立测试工具

`seg/` 目录包含独立的语义分割测试程序，不依赖 Car 框架：

```bash
cd seg && mkdir -p build && cd build && cmake .. && make
./seg_test <图片路径> [模型路径]
# 输出: 原图 | 分割掩码 | 叠加 三栏拼接图
```

## 同步文档

项目结构变更时需同步更新：
- `Car/Car/CLAUDE.md` (当前文件) — 开发指引
- `技术文档.md` — 知识库/简历用详细技术文档

## 标定

```bash
python tools/calibrate_perspective.py 图片路径 config/default.json
```

选四角点 → 输出尺寸 → 物理尺寸 → 预览鸟瞰 → 写入 calib 段。

## 项目开发规范

项目规范文件位于 `.claude/rules/` 目录下，按需读取（不要一次性全部加载）：

| 当你需要... | 读取这个文件 |
|------------|-------------|
| 了解项目全貌、入口、依赖 | [.claude/rules/项目认知报告.md](../.claude/rules/项目认知报告.md) |
| 设计模块架构、新增模块 | [.claude/rules/项目架构规范.md](../.claude/rules/项目架构规范.md) |
| 写代码、命名、注释 | [.claude/rules/编码规范.md](../.claude/rules/编码规范.md) |
| 提交代码、分支管理、构建 | [.claude/rules/开发流程规范.md](../.claude/rules/开发流程规范.md) |
| 写测试、检查清单 | [.claude/rules/测试规范.md](../.claude/rules/测试规范.md) |
| 查看所有规范概览 | [.claude/规范总索引.md](../.claude/规范总索引.md) |
