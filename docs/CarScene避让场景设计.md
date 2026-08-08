# CarScene 车辆避让场景设计

> 版本: v2.0 | 场景文件: `Car/Car/scene/scenes/CarScene.h/.cpp`
> 依赖改动: `PathFollower` 新增 `crop_scale_` 宽鸟瞰巡线

## 1. 场景职责

| 维度 | 说明 |
|------|------|
| 触发来源 | DetectionTask 检测到 `car` 框 → `scene_map["car"]="CarScene"` 自动切换进入 |
| 障碍类型 | **静止**车辆（停在赛道上） |
| 场景职责 | 进入后**全程沿"对侧边界"巡线**（障碍在左→沿右边界，障碍在右→沿左边界），不找中线 |
| 退出方式 | **完全交给 DetectionTask**：连续 `lost_frame_limit` 帧检测不到 car 自动切回 NormalScene |
| 决策原则 | 场景不管退出、不 hold、不 switchTo——一切交给 DetTask |

## 2. 为什么沿边界而不是偏移中线

| | 中线+偏移（旧方案） | 沿对侧边界（本方案） |
|---|---|---|
| 目标线 | 中线 + 参考点偏移 | 赛道真实边界 |
| 障碍污染 | 中线被障碍拉偏 | **对侧边界干净**（障碍在右时沿左边界，不受影响） |
| 复杂度 | 三状态机 + 斜坡 + 锁存 + 三层退出 | **一条 onCenterline 重写** |
| 车底位置 | 需理解"参考点偏移反转" | 车底中心参考在正中，目标线=边界 |

**"沿边界"的正确理解**：纯跟踪让车底中心收敛到目标路径上。目标路径=边界 → 车骑在边界上（半车出赛道）→ **出赛道不影响**，不需要向内平移让出量。

## 3. 核心流程

```
DetTask 检测到 car → 切入 CarScene
  ├─ 首帧：检测 car 框 → 底边中点 toBird → 与宽鸟瞰中心比较
  │   障碍在中线左 → edge=+1（沿右边界）；障碍在中线右 → edge=-1（沿左边界）
  │   方向只锁存一次（edge_locked_），之后不变（障碍静止）
  └─ 每帧：Follower.process(seg_mask, ctx)
         ├─ 宽鸟瞰（crop_scale_ = bird_width_scale）
         └─ onCenterline 重写：extractBoundary(mask, centers, edge) + smoothCenterline

DetTask 连续 lost_frame_limit 帧检测不到 car → 自动 switchTo("NormalScene")
```

**方向锁存**：切入 CarScene 第一帧（此时 DetTask 一定检测到了 car），框底边中点 vs 宽鸟瞰中心，锁存一次。障碍静止，方向不变。未锁存前沿中线（edge=0）兜底。

## 4. 坐标系

```
检测框（全图坐标）──减 frame_crop──▶ 裁切坐标 ──toBird()──▶ 宽鸟瞰坐标
```

- 检测框是全图坐标，先减 `frame_crop`（与 PersonScene 一致）
- `toBird()` 返回宽鸟瞰坐标
- 宽鸟瞰中心 x = `bird_width × bird_width_scale × 0.5`（= 车底中心参考，`onPurePursuit` 里 `w*0.5`）
- 框底边中点用 `y2`（地面接触点），框中心不在地面投影会偏近

## 5. PathFollower 改动（crop_scale_）

`PathFollower.h` 新增 protected 成员：

```cpp
float crop_scale_ = 1.0f;   // 巡线裁切宽度倍数（默认1=裁bird_width中心区）
```

`PathFollower.cpp` 裁切逻辑：

```cpp
int crop_w = static_cast<int>(calib.bird_width * crop_scale_);
```

CarScene 设 `setCropScale(bird_width_scale)` → 全宽鸟瞰（960px）。

**为什么宽鸟瞰**：沿边界走时赛道边界可能在裁切区之外（尤其弯道），全宽鸟瞰（960px vs 240px）保证远处边界始终在视野内，不丢线。**默认 crop_scale_=1 不影响其他场景。**

## 6. extractBoundary（PathPlanning.h）

新增泛化函数：

```cpp
// edge=-1 取左边界, +1 取右边界, 0 取中点
inline void extractBoundary(const cv::Mat& mask, std::vector<cv::Point2f>& out, int edge);
```

`extractCenterline` 改为 `extractBoundary(mask, out, 0)`，**其他场景零改动**。CarScene::Follower::onCenterline 用 `extractBoundary(mask, centers, edge)` 直接取边界点。

## 7. 退出（交给 DetTask）

DetectionTask 的切换逻辑（`DetectionTask.cpp`）：

```
在 CarScene（非 NormalScene）时，best_scene==NormalScene（car 框丢失）
  → det_lost_frames_++ → ≥ lost_frame_limit(40帧≈1.6s@25fps) → switchTo("NormalScene")
```

CarScene **不调用 hold/release/switchTo**，退出完全由 DetTask 驱动。

**注意**：切回 NormalScene 有横向跳变（边界→中线），`first_frame_=true` 继承 error/speed + 跳过 D 项可缓解，无脉冲。

## 8. HUD 显示

CarScene 运行期间，HUD 追加一行（仅 CarScene 显示，其他场景隐藏）：

```
Car: RIGHT edge / LEFT edge / CENTER
```

- `RIGHT edge`：沿右边界（障碍在左）｜`LEFT edge`：沿左边界（障碍在右）｜`CENTER`：未锁存沿中线
- 数据流：CarScene 首帧写 `ctx->car_edge`（原子量）→ DisplayTask HUD 读取
- 背景框在 CarScene 时加高（112→134px）容纳新行

## 9. config 参数

```json
"detection": { "scene_map": { "car": "CarScene" } },
"control": {
  "scene": {
    "CarScene": {
      "max_speed": 0.5,
      "pid_p": 0.95, "pid_d": 1.4, "aim_dist_near": 0.15,
      "coast_speed_decay": 0.9, "speed_recovery_k": 0.3
    }
  }
}
```

**不再需要** `avoid_trigger_time` / `avoid_offset_ratio`（旧方案参数，已删除）。CarScene 只用标准场景参数（max_speed / pid / aim_dist_near）。

## 10. 文件改动清单

| 文件 | 改动 |
|------|------|
| `Car/Car/scene/scenes/CarScene.h` | 新增：`Follower::onCenterline` 重写 + 首帧方向锁存 |
| `Car/Car/scene/scenes/CarScene.cpp` | 新增：`REGISTER_SCENE` + 锁存 + 沿边界巡线 |
| `Car/Car/scene/planning/PathFollower.h` | 加 `crop_scale_`（默认1） |
| `Car/Car/scene/planning/PathFollower.cpp` | 裁切用 `bird_width × crop_scale_` |
| `Car/Car/scene/planning/PathPlanning.h` | 加 `extractBoundary(mask,out,edge)`；`extractCenterline` 调它传0 |
| `Car/Car/thread/SharedContext.h` | 加 `car_edge` 原子量（CarScene写，DisplayTask读） |
| `Car/Car/tasks/DisplayTask.cpp` | 鸟瞰画布跟随 mask 原生宽 + HUD 沿边方向行 |
| `Car/Car/config/default.json` | `scene_map["car"]` + `scene["CarScene"]` |
| `Car/Car/CMakeLists.txt` | `add_executable` 加 `CarScene.cpp` |

不改 DetectionTask——退出逻辑已通用，配好 `scene_map` 即可。
