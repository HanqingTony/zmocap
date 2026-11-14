# zmocap

在 ComfyUI 里把**视频**变成**3D 骨骼动画**的节点包。

> 算法思想来自 [RGBMocap](https://github.com/)（ExMR / GVHMR / FootMR / KimodoMotionCorrection），
> 但**不是移植**——用 ComfyUI 自己的运行方式重新实现。

## 四条设计原则

1. **单环境**：只跑在 ComfyUI 自己的 Python 环境里。不建独立 venv、不用 pixi/comfy-env、不靠 subprocess 桥接。
2. **显存归 ComfyUI 管**：模型经 `comfy.model_management` 装载与回收，分块推理（默认 32 帧/块）。
3. **只出骨骼**：节点输出 SMPL-X 关节序列 `(T, J, 3)` 与参数；预览在节点内用 three.js 渲染，不依赖时间轴。
4. **用 ComfyUI 的方式重写**：视频解码用 VideoHelperSuite，批处理用 IMAGE 张量，缓存用节点执行缓存（不改它的项目/阶段系统）。

## 版本

当前 **0.2.0**：动作识别 + 预览已稳定（SMPL-X 链路与 RGBMocap 逐帧对照一致）。

| 版本 | 内容 |
|---|---|
| 0.1.0 | 环境自检、模型解析、YOLO 跟踪、ExMR 推理、NPZ/BVH 导出 |
| 0.2.0 | SMPL-X 原生网格预览、节点内 three.js 视图、骨骼小棍图层、VRM/SOMA 两条重定向线、预览器拆分 |

## 节点（18 个）

| 节点 | 作用 |
|---|---|
| `zmocap Environment` | 打印 python/torch/CUDA/模型目录 |
| `zmocap Models` | 解析 `models/zmocap`，报告缺件 |
| `zmocap Load Video` | 读视频为 IMAGE 批（PyAV，可指定采样 fps/截取帧数） |
| `zmocap Track` | YOLO 检测 + IoU 关联，选第 N 个人物 |
| `zmocap Frames` | IMAGE 批 → 帧目录（供引擎读盘） |
| `zmocap Intrinsics` | 估算相机内参 |
| `zmocap Sapiens2 Keypoints` | 448×336 关键点（body/hand/foot） |
| `zmocap SAM3D Body Features` | SAM3D 身体特征 |
| `zmocap HaMeR Hand Features` | HaMeR 手部特征 |
| `zmocap ExMR Solve` | 特征 → SMPL-X 参数 + 关节序列 |
| `zmocap SMPL Previewer` | 用官方 SMPL-X 网格预览（零重定向，作为对照基准） |
| `zmocap VRM Previewer` | 把 motion 绑到 VRM 角色并预览 |
| `zmocap VRM Loader` | 选择/上传角色文件 |
| `zmocap SMPL2SOMA` | SMPL-X → SOMA(78 关节) 通用运动 |
| `zmocap SOMA2VRM` | SOMA → 通用 humanoid motion |
| `zmocap Foot Contact` | 足部接触检测 |
| `zmocap Contact Load` | 载入已算好的接触数据 |
| `zmocap Export NPZ/BVH` | 导出 NPZ（关节 + 参数）与 BVH |

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
