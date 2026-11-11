# ComfyUI 现有 mocap 插件调研（2026-11）

目的：在动手写 zmocap 之前，确认 ComfyUI 生态里已经有哪些成熟做法，哪些直接借鉴、哪些明确避开。

| 项目 | ★ | 做法 | 借鉴 | 避开 |
|---|---:|---|---|---|
| [PozzettiAndrea/ComfyUI-MotionCapture](https://github.com/PozzettiAndrea/ComfyUI-MotionCapture) | 326 | GVHMR 全流程（ViTPose + HMR2 + SMPL），模型装到 `models/motion_capture`（~3.5 GB）；节点拆得细：`body_model/`（含免 torch 的 `smplx_lite` 手写 LBS）、`bvh_loader_node` / `bvh_viewer_node` / `bvh_retarget_node`、DVO 稠密位姿优化 | 节点命名与目录组织；BVH 查看/重定向节点；`install.py` 统一放模型 | `comfy-env`（pixi 隔离环境）——与"单环境"原则冲突 |
| [AKASubaz/ComfyUI-Video2MotionCapture](https://github.com/AKASubaz/ComfyUI-Video2MotionCapture) | 126 | GVHMR + **SAM3 分割跟踪**（不依赖 YOLO）+ 内嵌 Blender worker 做 FBX 导出与 Mixamo/UE 重定向 | SAM3 跟踪替代 YOLO 的思路；FBX/重定向需求清单 | 引入 Blender 子进程做几何后处理 |
| [lkevincc0/ComfyUI-GEM-X](https://github.com/lkevincc0/ComfyUI-GEM-X) | 8 | 视频 → YOLOX+ViTPose → SAM-3D-Body → GEM → **SOMA 77 关节** → 重定向 Mixamo/UE/Unity FBX；节点内 three.js 3D 预览 | 节点内 3D 预览的 `ui` payload + JS 实现；接触后处理的最小形式 | 自带 `.venv` + subprocess 后端 |
| [jtydhr88/ComfyUI-HY-Motion1](https://github.com/jtydhr88/ComfyUI-HY-Motion1) | 314 | 文生 3D 人体动作（HY-Motion 1.0） | 动作序列在 ComfyUI 里的张量约定 | — |
| [jtydhr88/ComfyUI-Kimodo](https://github.com/jtydhr88/ComfyUI-Kimodo) | 119 | 封装 nv-tlabs/kimodo（生成式动作） | 与"动作修正"互补的定位 | 与 RGBMocap 里的 KimodoMotionCorrection 不是同一项目 |
| [ameliacode/ComfyUI-SMPLx-Estimator](https://github.com/ameliacode/ComfyUI-SMPLx-Estimator) | 6 | 单图 → SMPL-X 参数，导出 obj/ply/glb | 单图侧输出格式 | 不做时序/世界坐标 |

## 结论

1. ComfyUI 里的 mocap 插件主流是 **GVHMR 系**（两个最大的都是），单图估计是 SMPL-X 系，GEM-X 走 SOMA 77 关节 + FBX 重定向。
2. **没有**插件把 ExMR（强足部/手部约束 + 轨迹级 ADMM 修正）做成节点——这正是 zmocap 的差异点。
3. 成熟插件的共同短板是**环境隔离**（comfy-env / 自带 venv / Blender 子进程）。zmocap 反其道：全部跑在 ComfyUI 自身环境里，靠 `comfy.model_management` 控显存。
4. 预览与导出已有成熟形态可抄：节点 `ui` payload + three.js（GEM-X）、BVH 查看器（MotionCapture）。
