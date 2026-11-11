"""ExMR 运行时胶水：构建资产目录、加载模型、跑三个阶段。

设计原则（见 docs/DESIGN.md）：全部在 ComfyUI 自己的 Python 环境里跑，
不建独立 venv；显存回收交给调用方（节点里用 comfy.model_management）。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

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
    import vendor_patch

    vendor_patch.patch(vendor, support, quiet=True)
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


class ExMRRunner:
    """按阶段加载/运行 ExMR 各组件；每个方法自己负责把模型搬上显存与卸载。"""

    def __init__(self, assets: Path, device: str = "cuda"):
        self.assets = Path(assets)
        self.device = device

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

    def features(self, frames: Path, boxes, intrinsics, keypoints=None, low_memory=True, hands=True):
        import torch
        from exmr.preprocessing.body_features import SamBodyFeatures

        body = SamBodyFeatures(
            str(self.assets / "sam-3d-body-dinov3"),
            batch_size=8,
            num_workers=0,
            low_memory=low_memory,
        )
        try:
            with torch.inference_mode():
                body_features = body(str(frames), torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(intrinsics))
        finally:
            del body
            _flush()
        if not hands:
            return body_features, None
        from exmr import pipeline as P
        from exmr.preprocessing.hand_features import HamerHandFeatures

        if keypoints is None:
            keypoints = self.keypoints(frames, boxes, low_memory=low_memory)
        hand_boxes = P.hand_boxes(keypoints, "sapiens2")
        hamer = HamerHandFeatures(
            str(self.assets / "hamer"), batch_size=8, num_workers=0, low_memory=low_memory
        )
        try:
            with torch.inference_mode():
                hand_features = hamer(str(frames), hand_boxes)
        finally:
            del hamer
            _flush()
        return body_features, hand_features

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
                    **{key: value[None].cuda() for key, value in data.items()},
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
