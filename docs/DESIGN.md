# zmocap 设计

## 目标

在 ComfyUI 里完成 RGBMocap 的核心能力：**视频 → 世界坐标 3D 骨骼动画**，但不搬它的 GUI、项目/阶段系统与多 runtime 安装器。

## 流水线（motion 之后是纯格式转换，角色只在最后出现）

```
LoadVideo → GetVideoComponents
  → Track / Frames / Models
  → Sapiens2 Keypoints / SAM3D Body Features / HaMeR Hand Features
  → ExMR Solve                 motion(smplx)
  → SMPL2SOMA                  motion(soma)      单进单出，不碰角色
  → SOMA2VRM                   motion(vrm)       单进单出，不碰角色
  → Rig ← VRM Loader           motion + 角色 → GLB（唯一需要角色的节点）
  → Export NPZ/BVH
```

约定：`motion["skeleton"]` ∈ {`smplx`, `soma`, `vrm`}；转换节点只换表示，绑定节点才碰角色。

## 绑定数学（与"增量叠加"的区别）

```
B_can   标准 humanoid 静止骨骼基（Y 沿骨向，T-pose，正面 -Z）
B_char  角色自己的静止骨骼基（可能是 A-pose、身高不同）
A(t)    运动在标准骨架上的世界旋转
W_char(t) = A(t) @ B_can⁻¹ @ B_char        （t 静止时 A=B_can ⇒ W_char=B_char）
```

方向由源关节位置驱动（`soma/vrm_motion.py`），因此 T-pose/A-pose、骨长与 roll 差异都不会把
骨链拧反；写入 GLB 时 `node_world = W(t) @ W_rest⁻¹ @ node_rest` 再按层级取 local。

## 流水线

```
VHS_LoadVideo ──► 人物跟踪 (YOLOX+BoT-SORT+OSNet)
                       │ tracks.json
                       ▼
                  裁剪批次 (IMAGE batch, 每 track 一段)
                       │
                       ▼
                  ExMR 骨骼 (Sapiens2 关键点 → ExMR → SMPL-X 参数)
                       │
            ┌──────────┴───────────┐
            ▼                      ▼
      足部锁定 (接触+ADMM+IK)   3D 预览 (节点内 three.js)
            │
            ▼
        导出 (BVH / NPZ)
```

## 为什么单环境可行

已在目标环境实测（ComfyUI 0.41.0 / Python 3.13.5 / torch 2.13.0+cu130）：

```
OK  exmr.model.exmr        OK  exmr.model.ik
OK  exmr.model.global_traj OK  exmr.geometry.camera
```

ExMR 核心模块在 ComfyUI 的 Python/torch 下可直接 import；缺的只是 pip 包（`ultralytics` / `smplx` / `PyTurboJPEG` / `yacs` / `chumpy`）与 vendored 的 `sapiens`。

## 显存策略

- 每个模型包一层轻量 wrapper，暴露 `model_size()` / `load()`，交给 `comfy.model_management`：
  - 装载：`comfy.model_management.load_models_gpu([model], memory_required=...)`
  - 回收：`comfy.model_management.soft_empty_cache()`
  - 设备：`comfy.model_management.get_torch_device()` / `unet_offload_device()`
- 分块推理：默认 32 帧/块，块间释放中间张量；README 里给出 `--lowvram` 下的阈值建议。

## 数据契约

| 类型 | 形状 | 说明 |
|---|---|---|
| `ZMOCAP_TRACKS` | JSON | 每人的 frame→bbox、track_id |
| `IMAGE` | `(B,H,W,3)` | 裁剪批次 |
| `ZMOCAP_SMPLX` | dict | `betas (10,)`、`global_orient (T,3,3)`、`transl (T,3)`、`body_pose (T,21,3,3)` |
| `ZMOCAP_JOINTS` | `(T,J,3)` | 世界坐标关节序列 |

## 里程碑

1. ✅ 仓库骨架 + 环境自检节点（可加载、可推到 ComfyUI）
2. 依赖补齐 + 权重就位 + ExMR 单段推理节点
3. 跟踪/裁剪节点（含多人）
4. 足部锁定 + 导出 BVH
5. 节点内 3D 预览（three.js）

## 命名

项目名 `zmocap`：`z` 系列命名（与 `z-llm`/`z-comfyui` 等一致），`mocap` 表能力。不再沿用上游项目名，避免与"移植"混淆。
