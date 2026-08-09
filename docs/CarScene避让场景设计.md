# 沿边界巡线 + CarScene 避让设计

> 版本: v4.0 | 涉及: PathFollower / PathPlanning / NormalScene / TrafficSignScene / CarScene
> 核心变化: **巡线从"跟中线"改为"沿边界"**，全程宽鸟瞰，**控制参数全局统一**

## 1. 总览

| 场景 | 沿边方向 |
|------|---------|
| NormalScene（默认） | **左边界**（`follow_edge_ = -1`） |
| TrafficSignScene | 直走→左边界，右转→右边界（`fork_side` 联动） |
| CarScene | 障碍在左→**右边界**，障碍在右→**左边界**（首帧锁存） |

**全局统一控制参数**（`ControlConfig`）：`max_speed` / `pid_p` / `pid_d` / `aim_dist_near` / `coast_speed_decay` / `speed_recovery_k`——所有场景一套，**场景切换无参数跳变**。场景只保留行为差异 `lateral_offset_ratio`（贴边偏移）。

## 2. 为什么沿边界而不是跟中线

| | 跟中线（旧） | 沿边界（新） |
|---|---|---|
| 目标线 | 左右边界中点 | 赛道真实边界（默认左） |
| 障碍污染 | 中线被障碍拉偏 | **对侧边界干净**（CarScene 用） |
| 岔路处理 | 需要 handleFork 切岔路 | **天然跟随最左/最右边界**，岔路由边界方向决定 |
| 车底参考 | 画面中心 | **画面中心（不变）** |

**关键理解**：
- 鸟瞰图是**固定变换 + 动态内容**：摄像头固定在车上，车底中心在鸟瞰图里**永远投影到画面中心** `(w*0.5, cy_ratio*h)`
- `onPurePursuit` 的参考点 `(cx, cy)` 就是这个画面中心，**不需要改**
- 沿边界走时变的是**边界线在画面里的位置**：车转向 → 左边界从画面左侧移向中心 → 车骑上边界
- 车底中心参考点 = 画面中心，**不是赛道中心**

## 3. PathFollower 核心改动

### 3.1 沿边方向成员

```cpp
// PathFollower.h
int follow_edge_ = -1;              // -1=左边界 +1=右边界 0=中点
void setFollowEdge(int edge);       // public，子类调用
```

### 3.2 全程宽鸟瞰（不再裁中心区）

```cpp
// PathFollower.cpp process()
cv::Mat wide_mask = ctx->perspective.warp(seg_mask);
cv::Mat mask = wide_mask;   // 全程宽鸟瞰，不再裁 240px 中心区
```

沿边界需要宽视野（边界可能远离画面中心），所有场景统一全宽鸟瞰（960px vs 原 240px 裁切）。

**岔路切除已停用**：`handleFork` 算法保留在 `PathPlanning.h` 备用，但 `process()` 不再调用 `onFork`。沿边界巡线天然跟随最左/最右边界，岔路选择由边界方向决定。如需恢复，在 `process()` 加回 `onFork(wide_mask, ctx)` 即可。

### 3.3 onCenterline 默认沿边界

```cpp
// PathFollower.cpp
void PathFollower::onCenterline(const cv::Mat& mask,
                                std::vector<cv::Point2f>& centers,
                                SharedContext* /*ctx*/) {
    pathplan::extractBoundary(mask, centers, follow_edge_);   // 默认左边界
    pathplan::smoothCenterline(centers);
}
```

`extractBoundary`（`PathPlanning.h`）：逐行扫描，`edge=-1` 取左边界、`+1` 右边界、`0` 中点。`extractCenterline` 保留为其传 0 的包装。

### 3.4 参考点横向偏移（场景参数）

车底中心参考点默认在画面中心（压线）。沿边巡线时，参考点可横向偏移，控制车贴边界内侧不压线：

```cpp
// PathFollower::onPurePursuit
float cx = w * (0.5f - follow_edge_ * sp.lateral_offset_ratio);
```

- `follow_edge_`：-1=左边界，+1=右边界
- `lateral_offset_ratio`：**每场景参数**（`SceneControlParams::lateral_offset_ratio`），默认 0=压线
- **贴边界内侧**：左边界 → 参考点偏右（cx 增大）；右边界 → 参考点偏左（cx 减小）——同一参数，符号自动决定
- 显示侧蓝点同步画到偏移后的参考点位置，且鸟瞰图垂直中心画红线区分左右

| 场景 | `lateral_offset_ratio` | `aim_dist_max` | `cx_smooth_frames` |
|------|------------------------|----------------|-------------------|
| NormalScene | 0.15（贴边内侧） | 0（不限） | 30（cx 平滑到位） |
| TrafficSignScene | 0.0（压线） | 0（不限） | 0（硬切） |
| CarScene | 0.1（贴边内侧） | 0.4（前视上限） | 0（硬切） |

- `aim_dist_max`：前视点最大距离 (m)，`0`=不限制。CarScene 沿边界走时限制前视，避免远处边界提取误差放大；`aim_dist = min(near + speed×k, aim_dist_max)`。
- `cx_smooth_frames`：车底中心参考点平滑到位帧数，`0`=硬切。NormalScene 从画面中心/当前值在 N 帧内线性平滑到目标偏移，避免场景切换的参考点横向跳变；其他场景硬切。平滑期间前视点保持当前速度下的动态前视（最远），参考点移动时用远前视稳定跟踪。

### 3.5 全局统一控制参数（场景切换无顿挫）

`PathFollower` 的 `onLost`/`onPD`/`onSpeed` 使用 `ControlConfig` **全局参数**；`onPurePursuit` 的前视距离/参考点偏移为全局参数，`lateral_offset_ratio` 为场景参数：

```cpp
// onPurePursuit：前视距离用全局
float aim_dist = ctrl.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;
// onPurePursuit：参考点偏移用场景参数
float cx = w * (0.5f - follow_edge_ * sp.lateral_offset_ratio);
// onPD：PID 用全局
float output = ctrl.pid_p * error + ctrl.pid_d * derivative;
// onSpeed：目标速度用全局
float target = ctrl.max_speed;
// onLost：盲开衰减用全局
last_speed_ *= ctrl.coast_speed_decay;
```

全局参数：`max_speed` / `pid_p` / `pid_d` / `aim_dist_near` / `coast_speed_decay` / `speed_recovery_k`。场景参数：`lateral_offset_ratio`（`SceneControlParams`）。

**为什么**：场景切换时若 PID/速度/前视参数整套变化，即使 `first_frame_` 继承 error/speed，控制系数突变 → PWM 输出跳变 → 顿挫。全局统一后切换只变 `follow_edge_`/`fork_side`/`lateral_offset_ratio`（行为），控制系数不变，平滑过渡。

`SceneControlParams` 精简为只有 `lateral_offset_ratio`（行为差异）。`TurnControlParams`（TrafficSignScene 右转专属速度/PID）已删除——全局统一后右转也用同一套参数。

## 4. CarScene（沿对侧边界避让）

```
DetTask 检测到 car → 切入 CarScene
  ├─ 实时检测：每帧 car 框底边中点 vs 赛道中心（在障碍 y 处插值赛道中线）
  │   障碍在赛道中左 → setFollowEdge(+1)（沿右边界）
  │   障碍在赛道中右 → setFollowEdge(-1)（沿左边界）
  │   每帧更新，不锁死（单帧检测有误差，实时纠正）
  └─ 每帧：Follower.process(seg_mask, ctx)   // 基类 onCenterline 用 follow_edge_
DetTask 连续 lost_frame_limit 帧检测不到 car → 自动切回 NormalScene
```

**判断基准 = 赛道中心，不是画面中心**：车沿边界走时，鸟瞰画面中心 ≠ 赛道中心。判断用 `trackCenterXAt(seg_mask, obs.y)`——宽鸟瞰提取中线、在障碍 y 处插值出赛道中心 x，比较 `obs.x < 赛道中心x`。中线为空时兜底用宽鸟瞰中心。

- **不重写 onCenterline**：直接用基类 `setFollowEdge` 切方向
- **实时检测不锁死**：每帧判断左右并 `setFollowEdge`，单帧误差（框抖动/插值误差）下一帧即纠正
- **不管退出**：不 hold、不 switchTo，一切交给 DetTask
- **车底中心**：参考点 = 画面中心 + 沿边方向×`lateral_offset_ratio`（§3.4），纯跟踪让车骑到偏移后的参考线上

## 5. TrafficSignScene（fork_side 联动边界）

```cpp
// TrafficSignScene.cpp
void TrafficSignScene::Follower::onCenterline(const cv::Mat& mask,
                                              std::vector<cv::Point2f>& centers,
                                              SharedContext* /*ctx*/) {
    pathplan::extractBoundary(mask, centers, (fork_side == 1) ? 1 : -1);
    pathplan::smoothCenterline(centers);
}
```

- 直走（fork_side=0）→ 左边界；右转（fork_side=1）→ 右边界
- `onFork`/`handleFork` 保留不动（岔路 mask 处理仍执行）
- 右转参数 `right.max_speed/pid` 保留

## 6. 坐标与显示

- 检测框（全图坐标）→ 减 `frame_crop` → `toBird()` → 宽鸟瞰坐标
- `bird_centers_` 现在画的是**边界点**（黄线 = 左边界），车底蓝点仍在**画面中心**——视觉正确
- 车底中心参考点 = 画面中心 `(w*0.5, cy_ratio*h)`，`onPurePursuit`/`onPD`/`onSpeed` 全不动

## 7. 文件改动清单

| 文件 | 改动 |
|------|------|
| `PathFollower.h` | 加 `follow_edge_=-1` + `setFollowEdge`；**删 `crop_scale_`** |
| `PathFollower.cpp` | process 用 wide_mask 全程宽鸟瞰；onCenterline 默认 `extractBoundary(follow_edge_)` |
| `PathPlanning.h` | `extractBoundary`（已存在）+ `extractCenterline` 包装传0 |
| `CarScene.h/.cpp` | 删 `onCenterline` 重写 + `setCropScale`；首帧锁存 `setFollowEdge` |
| `TrafficSignScene.h/.cpp` | Follower 加 `onCenterline` 重写（fork_side→edge）；onFork 保留 |
| `SharedContext.h` | `car_edge` 原子量（CarScene写，DisplayTask读） |
| `DisplayTask.cpp` | 鸟瞰画布跟随 mask 原生宽 + HUD 沿边方向行 |
| `config/default.json` | `scene_map["car"]` + `scene["CarScene"]` |
