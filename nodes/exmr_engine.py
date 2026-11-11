"""ExMR 运行时胶水：构建资产目录、加载模型、跑三个阶段。

设计原则（见 docs/DESIGN.md）：全部在 ComfyUI 自己的 Python 环境里跑，
不建独立 venv；显存回收交给调用方（节点里用 comfy.model_management）。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

# 模型根目录下的相对路径（用户把权重放在 ComfyUI/models/zmocap/）
SOURCES = {
    "exmr": "exmr/exmr.safetensors",
    "smplx": "body_models/smplx/SMPLX_NEUTRAL.npz",
    "sapiens2": "exmr/sapiens2_1b_pose.safetensors",
    "flow_sam": "flowhmr/sam/model.ckpt",
    "flow_sam_config": "flowhmr/sam/model_config.yaml",
    "flow_mhr": "flowhmr/sam/assets/mhr_model.pt",
    "hand_hamer": "handflow/hamer.ckpt",
    "hand_mean": "handflow/mano_mean_params.npz",
}

# ExMR 代码内部期望的 assets 布局
TARGETS = {
    "exmr": "exmr/exmr.safetensors",
    "smplx": "body_models/smplx/SMPLX_NEUTRAL.npz",
    "sapiens2": "sapiens2/sapiens2_1b_pose.safetensors",
    "flow_sam": "sam-3d-body-dinov3/model.ckpt",
    "flow_sam_config": "sam-3d-body-dinov3/model_config.yaml",
    "flow_mhr": "sam-3d-body-dinov3/assets/mhr_model.pt",
    "hand_hamer": "hamer/hamer_ckpts/checkpoints/hamer.ckpt",
    "hand_mean": "hamer/data/mano_mean_params.npz",
}

# 目录级链接：vendored 代码按目录找 MANO 模型
DIR_SOURCES = {
    "hamer/data/mano": "body_models/mano",
}

_READY = False


def _linked(source: Path, target: Path) -> bool:
    try:
        return target.exists() and target.samefile(source)
    except OSError:
        return False


def build_assets(models_root: Path, vendor: Path, assets: Path | None = None) -> Path:
    """把权重软链成 ExMR 期望的 assets 布局，并准备 Python 路径/hub 补丁。"""
    global _READY
    models_root = Path(models_root)
    vendor = Path(vendor)
    assets = Path(assets) if assets else models_root / "_assets"
    for name in ("exmr", "third_party/sapiens2/sapiens", "third_party/hamer/hamer", "third_party/dinov3"):
        if not (vendor / name).is_dir():
            raise RuntimeError(f"ExMR 代码缺失: {vendor / name}（vendor/ExMR 应随仓库部署）")
    missing = [rel for rel in SOURCES.values() if not (models_root / rel).is_file()]
    if missing:
        raise FileNotFoundError("模型缺失，请放到 models/zmocap/：" + ", ".join(missing))

    support = Path(__file__).resolve().parent.parent / "support"
    for path in (vendor, vendor / "third_party/sapiens2", vendor / "third_party/hamer", support):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # 让 vendored 代码用我们自己的助手，而不是原程序的私有包
    import chumpy_compat
    import vendor_patch

    vendor_patch.patch(vendor, support, quiet=True)
    chumpy_compat.ensure()  # MANO/SMPL .pkl 需要 chumpy，这里在 Python 3.13 上补兼容
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    assets.mkdir(parents=True, exist_ok=True)
    for key, target_rel in TARGETS.items():
        source = models_root / SOURCES[key]
        target = assets / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if _linked(source, target):
            continue
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            target.symlink_to(source)
        except OSError as exc:  # Windows 需要开发者模式
            raise RuntimeError(f"无法创建模型链接 {target} → {source}：{exc}") from exc

    for target_rel, source_rel in DIR_SOURCES.items():
        source = models_root / source_rel
        target = assets / target_rel
        if not source.is_dir():
            raise FileNotFoundError(f"模型目录缺失，请放到 models/zmocap/：{source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if _linked(source, target):
            continue
        if target.is_symlink() or target.exists():
            if target.is_dir() and not target.is_symlink():
                raise RuntimeError(f"{target} 已是真实目录，请先移走")
            target.unlink()
        target.symlink_to(source, target_is_directory=True)

    config = vendor / "configs/hamer_model_config.yaml"
    shutil.copy2(config, assets / "hamer/hamer_ckpts/model_config.yaml")

    if not _READY:
        original = torch_hub_load()

        def local_hub(repository, *args, **kwargs):
            if repository == "facebookresearch/dinov3":
                repository = str(vendor / "third_party/dinov3")
                kwargs["source"] = "local"
            return original(repository, *args, **kwargs)

        import torch

        torch.hub.load = local_hub
        _READY = True
    return assets


def torch_hub_load():
    import torch

    return torch.hub.load


def frame_intrinsics(intrinsics, count: int):
    """把共享的 3x3 内参扩成逐帧 (count, 3, 3)，vendored 代码要求逐帧。

    tracking.npz 里只存一份 3x3（同一台机器同一段视频），ExMR 的
    _FrameDataset 却断言 len(boxes) == len(K)，所以这里对齐。
    """
    import numpy as np

    k = np.asarray(intrinsics, dtype=np.float32)
    if k.shape == (3, 3):
        return np.repeat(k[None], int(count), axis=0)
    if k.ndim == 3 and k.shape[1:] == (3, 3) and k.shape[0] == int(count):
        return k
    raise ValueError(f"内参形状 {k.shape} 无法对齐 {count} 帧")


class ExMRRunner:
    """按阶段加载/运行 ExMR 各组件；每个方法自己负责把模型搬上显存与卸载。"""

    def __init__(self, assets: Path, device: str = "cuda"):
        self.assets = Path(assets)
        self.device = device

    # --- 组装模型输入 -----------------------------------------------------
    def model_inputs(self, keypoints, boxes, intrinsics, rotations=None, body=None, hands=None, backend="sapiens2"):
        """把 numpy 输入整理成 vendored pipeline 要的张量（逐帧内参已对齐）。"""
        import numpy as np
        import torch
        from exmr import pipeline as P

        count = int(np.asarray(boxes).shape[0])
        if rotations is None:
            rotations = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
        return P.model_inputs(
            torch.as_tensor(np.asarray(keypoints), dtype=torch.float32),
            torch.as_tensor(np.asarray(boxes), dtype=torch.float32),
            torch.as_tensor(frame_intrinsics(intrinsics, count)),
            torch.as_tensor(np.asarray(rotations), dtype=torch.float32),
            body,
            hands,
            backend,
        )

    # --- 三个阶段 ---------------------------------------------------------
    def keypoints(self, frames: Path, boxes, low_memory=True):
        import torch
        from exmr.preprocessing.keypoints.sapiens2 import Sapiens2Keypoints

        model = Sapiens2Keypoints(
            str(self.assets / "sapiens2/sapiens2_1b_pose.safetensors"),
            resolution="448x336",
            num_workers=0,
            low_memory=low_memory,
        )
        try:
            with torch.inference_mode():
                return model(str(frames), torch.as_tensor(boxes, dtype=torch.float32))
        finally:
            del model
            _flush()

    def body_features(self, frames: Path, boxes, intrinsics, low_memory=True):
        """SAM 3D Body 的 body decoder pose token，每帧一个 (L, 1024)。"""
        import torch
        from exmr.preprocessing.body_features import SamBodyFeatures

        model = SamBodyFeatures(
            str(self.assets / "sam-3d-body-dinov3"),
            batch_size=8,
            num_workers=0,
            low_memory=low_memory,
        )
        try:
            with torch.inference_mode():
                count = int(torch.as_tensor(boxes).shape[0])
                aligned = frame_intrinsics(intrinsics, count)
                return model(
                    str(frames),
                    torch.as_tensor(boxes, dtype=torch.float32),
                    torch.as_tensor(aligned),
                )
        finally:
            del model
            _flush()

    def hands(self, frames: Path, boxes, intrinsics, keypoints=None, low_memory=True, backend="sapiens2"):
        """HaMeR 手部 token（每帧左右手各一个特征）。"""
        import torch
        from exmr import pipeline as P
        from exmr.preprocessing.hand_features import HamerHandFeatures

        if keypoints is None:
            keypoints = self.keypoints(frames, boxes, low_memory=low_memory)
        hand_boxes = P.hand_boxes(torch.as_tensor(np.asarray(keypoints)), backend)
        model = HamerHandFeatures(
            str(self.assets / "hamer"), batch_size=8, num_workers=0, low_memory=low_memory
        )
        try:
            with torch.inference_mode():
                return model(str(frames), hand_boxes)
        finally:
            del model
            _flush()

    def features(self, frames: Path, boxes, intrinsics, keypoints=None, low_memory=True, hands=True):
        body_features = self.body_features(frames, boxes, intrinsics, low_memory=low_memory)
        if not hands:
            return body_features, None
        return body_features, self.hands(
            frames, boxes, intrinsics, keypoints=keypoints, low_memory=low_memory
        )

    def infer(self, data, static_cam=True, fps=30.0, smooth_global=True):
        import torch
        from exmr.model.exmr import ExMR

        model = ExMR.from_pretrained(
            str(self.assets / "exmr/exmr.safetensors"),
            str(self.assets / "body_models/smplx/SMPLX_NEUTRAL.npz"),
        )
        try:
            with torch.inference_mode():
                output = model(
                    **{
                        key: torch.as_tensor(np.asarray(value, dtype=np.float32))[None].cuda()
                        for key, value in data.items()
                        if value is not None
                    },
                    static_cam=static_cam,
                    snap_to_ground=False,
                    smooth_incam=False,
                    smooth_global=smooth_global,
                    fps=fps,
                    normalize_world=False,
                )
                native = {
                    space: {key: value[0].cpu() for key, value in output[space].items()}
                    for space in ("smplx_incam", "smplx_global")
                }
                native["static_conf_logits"] = output["static_conf_logits"][0].cpu()
                native["fps"] = fps
        finally:
            del model
            _flush()
        return native


def _flush():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
