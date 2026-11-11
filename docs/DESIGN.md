# zmocap 设计

## 目标

在 ComfyUI 里完成 RGBMocap 的核心能力：**视频 → 世界坐标 3D 骨骼动画**，但不搬它的 GUI、项目/阶段系统与多 runtime 安装器。

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
