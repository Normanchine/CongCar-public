# NormalScene 巡线设计

> 版本: v1.0 | 涉及: `NormalScene` / `PathFollower` / `PathPlanning`
> 核心: **沿边界巡线（边界由全局继承）+ 纯跟踪 + PD + 丢线盲开**

## 1. 总览

NormalScene 是默认场景（赛道正常行驶），本身**几乎没有逻辑**——它只是 `PathFollower` 的一个薄封装：

```cpp
// NormalScene.cpp
void NormalScene::run(SharedContext* ctx) {
    auto seg = ctx->seg_result.get();
    if (!seg || seg->mask.empty()) return;
    follower_.process(seg->mask, ctx);   // 全部巡线逻辑在 PathFollower 里
}
```

真正的巡线逻辑在 `PathFollower::process`（模板方法），本场景用**全部默认实现**，无任何重写。

## 2. 每帧流程（PathFollower::process）

```
seg_mask（分割结果）
   │
   ▼ 0. 透视变换
wide_mask = perspective.warp(seg_mask)          // 宽鸟瞰（960×480）
   │
   ▼ 1. 全程宽鸟瞰（不裁中心区）
mask = wide_mask
   │
   ▼ 2. 边界提取（onCenterline 默认）
centers = extractBoundary(mask, centers, follow_edge_)   // 沿边方向（首帧从全局继承）
centers = smoothCenterline(centers)                      // 滑动平滑
   │
   ├─ centers 为空？ ──► onLost（盲开/停车保护）
   │
   ▼ 3. 纯跟踪（onPurePursuit 默认）
aim_pt = findLookAhead(centers, cx, cy, aim_dist)   // 沿边界找前视点
error  = headingError(cx, cy, aim_pt)               // 航向误差
   │
   ▼ 4. PD 控制（onPD 默认）
pwm = pwm_mid - (pid_p*error + pid_d*derivative) * pwm_k   // 钳到 [pwm_min, pwm_max]
// pid_p/pid_d 为左边界；右边界用 pid_p_right/pid_d_right（onPD 按 follow_edge_ 选）
   │
   ▼ 5. 速度控制（onSpeed 默认）
target = max_speed（大弯减速 + 加速平滑恢复）
   │
   ▼ 6. 写回共享状态
steering_pwm / target_speed / last_error
```

## 3. 边界提取（沿边巡线）

`PathFollower::onCenterline` 默认实现：

```cpp
pathplan::extractBoundary(mask, centers, follow_edge_);   // -1=左边界
pathplan::smoothCenterline(centers);
```

### 3.1 extractBoundary（PathPlanning.h）

逐行扫描，每行找赛道左右白色边界，输出指定侧：

```cpp
// edge=-1 取左边界, +1 取右边界, 0 取中点
for (每行 r):
    left  = 该行第一个白像素 x
    right = 该行最后一个白像素 x
    x = (edge<0) ? left : (edge>0) ? right : (left+right)/2
    out.push({x, r})
```

**NormalScene 的 `follow_edge_` 首帧从全局继承**（默认左，CarScene 改写后沿用），不再固定左边界。输出点序：行 r 从 0（鸟瞰顶部=远）到 h-1（底部=近车）。

### 3.2 smoothCenterline（PathPlanning.h）

滑动平均滤波（默认窗口 5），**只平滑偏移方向、保留扫描坐标**：
- 纵向赛道（Y 跨度 > X 跨度）→ 每点 x 取邻域均值，保留行号
- 横向赛道（X 跨度 > Y 跨度）→ 每点 y 取邻域均值，保留列号

## 4. 参考点：目标偏移直接生效

车底中心参考点 `(cx, cy)` 是纯跟踪的锚点，横向偏移直接取目标值（边界已全局化，场景切换不跳变，无需平滑）：

```cpp
// onPurePursuit
float target_cx_ratio = -follow_edge_ * sp.lateral_offset_ratio;  // 目标偏移
float cx = w * (0.5f + target_cx_ratio);   // 画面中心 + 目标偏移
float cy = h * ctrl.car_cy_ratio;          // 纵向固定 95% 高度
```

### 4.1 目标偏移的方向语义

| follow_edge_ | lateral_offset_ratio | cx | 效果 |
|-------------|---------------------|----|------|
| -1（左边界） | 0.15 | `w*(0.5+0.15)=0.65w` | 参考点偏右 → 车贴左边界内侧 |
| +1（右边界） | 0.15 | `w*(0.5-0.15)=0.35w` | 参考点偏左 → 车贴右边界内侧 |

**为什么直接生效**：边界由全局 `follow_edge_global` 继承，场景切换回 Normal 沿用同一边界，参考点目标不跳变，无需平滑过渡。

## 5. 纯跟踪（onPurePursuit 默认）

```cpp
float cur_spd = (last_speed_ > 0.01f) ? last_speed_ : ctrl.max_speed;
float aim_dist = ctrl.aim_dist_near + cur_spd * ctrl.lookahead_speed_k;  // 动态前视
if (sp.aim_dist_max > 0 && aim_dist > sp.aim_dist_max) aim_dist = sp.aim_dist_max;

findLookAhead(centers, cx, cy, aim_dist, ppm_x, ppm_y, aim_pt);  // 找前视点
error = headingError(cx, cy, aim_pt.x, aim_pt.y, car_length, ppm_x, ppm_y) * steering_dir;
```

### 5.1 findLookAhead（PathPlanning.h）

1. 找离 `(cx, cy)` 最近的中线点作为弧长起点
2. 从该点沿边界**向前**（远处）累加弧长（ppm_x/ppm_y 分离）
3. 累加距离 ≥ `aim_dist` 的点即前视点

### 5.2 headingError（PathPlanning.h）

标准纯跟踪模型：

```
dx = (aim_x - cx) / ppm_x
dy = (cy - aim_y) / ppm_y + car_length
error = -atan(2 * car_length * dx / (dx²+dy²))
```

## 6. PD 控制（onPD 默认）

```cpp
float derivative = first_frame_ ? 0 : (error - last_error_);  // 首帧跳 D（防场景切换脉冲）
output = (follow_edge_ > 0 ? pid_p_right : pid_p) * error       // 左/右边界分开调 PD
       + (follow_edge_ > 0 ? pid_d_right : pid_d) * derivative;
pwm = pwm_mid - output * pwm_k;
return clamp(pwm, pwm_min, pwm_max);
```

## 7. 速度控制（onSpeed 默认）

```cpp
float target = ctrl.max_speed;
if (|error| > steer_speed_threshold)                      // 大弯减速
    target = ctrl.max_speed / (1 + steer_speed_k * (|error| - threshold));
if (target < last_speed_) return target;                  // 减速立即生效
return last_speed_ + (target - last_speed_) * speed_recovery_k;   // 加速平滑恢复
```

## 8. 丢线容错（onLost 默认）

`centers` 为空（边界提取不到）时进入：

```cpp
lost_frames_++;
if (lost_frames_ > max_lost_frames) { stop = true; return; }   // 超限停车保护

step = coast_steer_step * lost_frames_ / max_lost_frames;      // 逐帧递增偏移
pwm  = clamp(last_pwm_ + steering_dir * step, pwm_min, pwm_max);  // 盲开只右转+渐偏
last_speed_ = coast_speed;                                     // 盲开固定速度
```

| 阶段 | 行为 |
|------|------|
| 短时丢失（≤max_lost_frames） | 盲开：**只右转**（赛道在右侧）+ 逐帧递增偏移 + **固定速度（coast_speed）** |
| 彻底丢失（>max_lost_frames） | 停车保护 `stop=true` |

## 9. config 相关参数（NormalScene）

```json
"control": {
  "max_speed": 1.3, "pid_p": 0.3, "pid_d": 1.5,
  "pid_p_right": 0.3, "pid_d_right": 1.5, "aim_dist_near": 0.15,
  "car_cy_ratio": 0.95, "car_length": 0.193,
  "pwm_mid": 1560, "pwm_min": 500, "pwm_max": 2500, "pwm_k": 600,
  "steer_speed_threshold": 0.15, "steer_speed_k": 1.5,
  "lookahead_speed_k": 0.15, "max_lost_frames": 30,
  "coast_steer_step": 200, "coast_speed": 0.5, "speed_recovery_k": 0.01,
  "scene": {
    "NormalScene": { "lateral_offset_ratio": 0.15 }
  }
}
```

| 参数 | 作用 | NormalScene 值 |
|------|------|---------------|
| `lateral_offset_ratio` | 参考点横向偏移（贴边内侧） | 0.15 |
| `aim_dist_max` | 前视上限（NormalScene 不限） | 0 |

## 10. 显示

- 黄线：`bird_centers_`（当前边界线，NormalScene=左边界）
- 红点：`bird_aim_pt_`（前视点）
- 蓝点：车底参考点（画面中心 + 目标偏移）
- 红线：鸟瞰垂直中心分界线

## 11. 与其他场景的关系

| 场景 | 沿边方向 | 前视上限 |
|------|---------|---------|
| NormalScene（默认） | 全局继承（默认左） | 无 |
| TrafficSignScene | fork_side 联动（0→左/1→右） | 无 |
| CarScene | 实时检测（障碍对侧） | 0.4 |
