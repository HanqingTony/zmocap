# zmocap

在 ComfyUI 里把**视频**变成**3D 骨骼动画**的节点包。

> 算法思想来自 [RGBMocap](https://github.com/)（ExMR / GVHMR / FootMR / KimodoMotionCorrection），
> 但**不是移植**——用 ComfyUI 自己的运行方式重新实现。

## 四条设计原则

1. **单环境**：只跑在 ComfyUI 自己的 Python 环境里。不建独立 venv、不用 pixi/comfy-env、不靠 subprocess 桥接。
2. **显存归 ComfyUI 管**：模型经 `comfy.model_management` 装载与回收，分块推理（默认 32 帧/块）。
3. **只出骨骼**：节点输出 SMPL-X 关节序列 `(T, J, 3)` 与参数；预览在节点内用 three.js 渲染，不依赖时间轴。
4. **用 ComfyUI 的方式重写**：视频解码用 VideoHelperSuite，批处理用 IMAGE 张量，缓存用节点执行缓存（不改它的项目/阶段系统）。

## 节点（规划）

| 节点 | 作用 | 状态 |
|---|---|---|
| `zmocap · 环境自检` | 打印 python/torch/CUDA/模型目录 | ✅ |
| `zmocap · 模型` | 解析 `models/zmocap`，报告缺件 | ✅ |
| `zmocap · 人物跟踪` | YOLO 检测 + IoU 关联（多人物里选第 N 个） | ✅ 首版 |
| `zmocap · ExMR 骨骼` | Sapiens2 关键点 → SAM Body/HaMeR 特征 → ExMR → SMPL-X + 关节 | ✅ 首版 |
| `zmocap · 3D 预览` | 节点内 canvas 投影渲染（可拖动旋转、点击暂停） | ✅ |
| `zmocap · 导出` | NPZ（关节 + 参数）+ BVH | ✅ 首版 |
| `zmocap · 足部锁定` | 接触检测 + ADMM + 接触 IK 修正 | 待实现 |

模型布局（`ComfyUI/models/zmocap/`）：

```
exmr/exmr.safetensors                 exmr/sapiens2_1b_pose.safetensors
body_models/smplx/SMPLX_NEUTRAL.npz   flowhmr/sam/{model.ckpt,model_config.yaml,assets/mhr_model.pt}
handflow/hamer.ckpt                   handflow/mano_mean_params.npz
yolo/yolov8x.pt
```

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HanqingTony/zmocap.git
cd zmocap && pip install -r requirements.txt
```

依赖装进 **ComfyUI 自己的环境**，不额外建环境。

## 模型

放到 `ComfyUI/models/zmocap/`，用下面的命令自检：

```bash
python install.py            # 默认 <ComfyUI>/models/zmocap
python install.py --models-dir /path/to/models/zmocap
```

## 目录

```
zmocap/
├── __init__.py          # ComfyUI 入口
├── nodes/               # 节点实现
├── web/js/              # 前端扩展（骨骼预览）
├── docs/DESIGN.md       # 架构与里程碑
├── docs/RESEARCH.md     # ComfyUI 现有 mocap 插件调研
├── install.py           # 模型自检
└── requirements.txt
```

## 致谢

- [PozzettiAndrea/ComfyUI-MotionCapture](https://github.com/PozzettiAndrea/ComfyUI-MotionCapture)（节点组织、BVH 查看/重定向思路）
- [AKASubaz/ComfyUI-Video2MotionCapture](https://github.com/AKASubaz/ComfyUI-Video2MotionCapture)（SAM3 跟踪、FBX 重定向思路）
- [lkevincc0/ComfyUI-GEM-X](https://github.com/lkevincc0/ComfyUI-GEM-X)（节点内 3D 预览）
- ExMR / GVHMR / FootMR / KimodoMotionCorrection 原作者

## 许可

本项目代码 MIT。vendored 的第三方代码保留其原始许可，见 `NOTICE`。
