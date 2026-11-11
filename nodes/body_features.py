"""SAM 3D Body 特征节点（阶段 2）：每帧一个 body decoder pose token。

自包含实现：直接用 vendored `exmr.preprocessing.body_features.SamBodyFeatures`
（它用 forward hook 取 decoder 的 pose token，并把 `_mocap_features_only` 打开），
不依赖任何第三方 SAM3DBody 节点包。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .exmr_engine import ExMRRunner, build_assets


class ZMocapSam3DBodyFeatures:
    """SAM 3D Body 的 (T, 1024) pose token，供 ExMR 当图像特征。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("ZMOCAP_FRAMES",),
                "tracks": ("ZMOCAP_TRACKS",),
                "models": ("ZMOCAP_MODELS",),
            },
            "optional": {"low_memory": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("ZMOCAP_BODY_FEATURES",)
    RETURN_NAMES = ("body_features",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, frames, tracks, models, low_memory=True):
        assets = build_assets(Path(models["root"]), Path(models["vendor"]))
        count = int(frames["count"])
        boxes = np.asarray(tracks["boxes"], dtype=np.float32)[:count]
        intrinsics = np.asarray(tracks["K"], dtype=np.float32)[:count]
        runner = ExMRRunner(assets)
        features = runner.body_features(
            Path(frames["dir"]), boxes, intrinsics, low_memory=bool(low_memory)
        )
        array = np.asarray(features.detach().cpu() if hasattr(features, "detach") else features)
        print(f"[zmocap] sam3d body features {array.shape}")
        return ({"values": array.astype(np.float32), "count": count},)
