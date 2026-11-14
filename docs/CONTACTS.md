# 脚接触数据（zmocap.contact.v2）

第一版：**自动检测 + 文件输出**。不做 ComfyUI 内时间轴；需要手改时编辑 JSON，
再用 `ZMocapContactLoad` 读回来。

## 节点

```
ZMocapExMR.motion
  → ZMocapFootContact
       ├─ motion    （原样透传；ExMR 已经做过 static_joint_transl / static_joint_ik / put_on_ground）
       └─ contacts  → ZMocapExport.contacts
                       → <filename>.contacts.json
                       → <filename>.contacts.npz
```

`ZMocapFootContact` 的可调参数：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `static_conf` | 0.8 | `static_conf_logits` 的 sigmoid 阈值，左右脚各取踝/脚两通道最大值 |
| `height_tolerance` | 0.05 m | 脚底最低点 ≤ 全身最低点 + 该值，才算“贴地” |
| `speed_limit` | 0.7 m/s | 脚踝/脚关节的 XZ 速度上限，过滤“低但在滑” |
| `vertical_limit` | 0.8 m/s | 脚底垂直速度上限 |
| `min_frames` | 2 | 短于该帧数的接触段丢弃 |
| `gap_frames` | 2 | 两侧都在接触的内部短暂断开，补回 |

算法沿用 ExMR 自己的定义（`static_conf` + 脚底最低点），并加了 RGBMoCap 风格的
速度门限、最短支撑、补洞。不是重新训练模型。

## JSON（人类可读、可手改）

文件名：`<motion 文件名>.contacts.json`

```json
{
  "format": "zmocap.contact.v2",
  "fps": 24.0,
  "frames": 124,
  "space": { "frame": "world", "up": "Y", "unit": "m" },
  "floor": { "mode": "plane", "height": 0.0 },
  "feet": {
    "left": {
      "ranges": [[12, 47], [60, 88]],
      "events": [
        { "frame": 12, "type": "contact_start" },
        { "frame": 47, "type": "contact_end" }
      ]
    },
    "right": { "ranges": [[0, 11], [47, 60], [88, 124]] }
  },
  "meta": {
    "source": "auto",
    "algorithm": "exmr_static_conf+foot_height",
    "thresholds": {
      "static_conf": 0.8,
      "height_tolerance": 0.05,
      "speed_limit": 0.7,
      "vertical_limit": 0.8,
      "min_frames": 2,
      "gap_frames": 2
    },
    "version": "1.0.0",
    "motion_sha256": "…"
  }
}
```

约定：

- 帧号 **0 起**，区间 **半开 `[start, end)`**。
- 坐标系：world、Y 向上、米。
- `floor.height` 是检测到的地面高度；v1 只支持单层平面。
- `source`：`auto` / `manual` / `mixed`。
- `motion_sha256` 绑定动作；动作变了 contact 就不再对应当前 motion。
- `ranges` 是主数据；逐帧数组在 NPZ。

## NPZ（给数值消费）

文件名：`<motion 文件名>.contacts.npz`

| key | 形状 | 说明 |
|---|---|---|
| `contact` | `(T,2) uint8` | 0/1，列顺序 left/right |
| `probability` | `(T,2) float32` | 左右脚置信度（sigmoid 后的 max） |
| `height` | `(T,2) float32` | 左右脚底最低点高度（米） |
| `fps` | scalar | |
| `frames` | scalar | |

`ZMocapExport` 也会把 `contact` / `contact_probability` / `contact_height` 冗余写进
`<filename>.npz`，方便只读一个文件；`static_conf_logits` 也会被保存，便于以后重新检测。

## 手改流程

1. 跑一次，得到 `<filename>.contacts.json`。
2. 编辑 `feet.left.ranges` / `feet.right.ranges`；保持 `[start, end)`、0 起。
3. 加 `ZMocapContactLoad` 节点，`path` 指向改后的 JSON；输出 `contacts`。
4. 把 `ZMocapContactLoad.contacts` 接到 `ZMocapExport.contacts`；`motion` 仍从
   `ZMocapFootContact.motion` 或 `ZMocapExMR.motion` 来。
5. `ZMocapContactLoad` 会把 `meta.source` 标成 `manual`；如果和 motion 的
   `motion_sha256` 不一致会打印警告（允许手动覆盖）。

## v1 实测

`zmocap_test.mp4`（124 帧 @ 24 fps，480×864）跑完整 ExMR 链路：

- 左脚 range `[[0, 124)]`，右脚 `[]`；
- 左脚踝高度 0.07 m，右脚踝 0.20–0.32 m；右脚底最低点比左脚高 0.12–0.19 m；
- 右脚全程离地，检测正确（不是漏检）。

这条视频本身是单脚支撑的姿势；换成走路/跑步视频，左右脚应该交替出现区间。

## v1 的边界

- 只检测左右脚整体接触，不区分脚跟/前掌/脚趾，不做手接触。
- 单层平面；楼梯/斜坡以后用 `floor.mode="segments"` 扩展。
- 自动检测没有人工时间轴；手改走 JSON。
- 修正动作沿用 ExMR 内置的 static-joint 修正；没有额外做脚锁 IK。
  如果后面要做“按 contact ranges 的强制脚锁”，再加 `Foot Lock` 节点消费 `ranges`。
- 没有引擎适配器；Unreal/Unity/Blender 的 Notify/Marker 转换以后按需加。

## v2：多通道

`channels` 决定 `arrays.contact` 的列顺序，每个通道都能在节点上单独开关：

| 通道 | 判定部位 | 默认 | 颜色 | 置信度来源 |
|---|---|---|---|---|
| `foot_left` / `foot_right` | 脚 | 开 | 蓝 `#3b82f6` / 红 `#ef4444` | ExMR `static_conf_logits` |
| `hand_left` / `hand_right` | 腕 + 五指 | 关 | 绿 `#22c55e` / 橙 `#f59e0b` | 几何 |
| `knee_left` / `knee_right` | 髋+膝 | 关 | 紫 `#a855f7` / 粉 `#ec4899` | 几何 |
| `pelvis` | 骨盆 | 关 | 黄 `#eab308` | 几何 |
| `head` | 颈/头/下颌/眼 | 关 | 白 `#e5e7eb` | 几何 |

几何判定 = 「该通道最低顶点离**当帧全身最低点**的距离 ≤ `height_tolerance`」
且水平速度 < `speed_limit` 且垂直速度 < `vertical_limit`；
脚的置信度换成 `sigmoid(static_conf_logits) ≥ static_conf`。
判定后统一做 `min_frames` / `gap_frames` 区间形态学。

场景里永远只有一个水平地面：`floor.height` 取全场最低点，`arrays.height` 是
**相对当帧地面**的高度（所以 0 表示贴地），`arrays.level` 是逐帧地面高度。
