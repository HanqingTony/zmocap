"""HaMeR 手部特征节点（阶段 3）：帧 → 左右手 token。

手部框来自 Sapiens2 关键点（`exmr.pipeline.hand_boxes`），所以这个节点要么连
关键点输出，要么自己在内部复用（`keypoints` 可选输入）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .exmr_engine import ExMRRunner, build_assets


class ZMocapHaMeRHandFeatures:
    """HaMeR 提取左右手特征，交给 ExMR 解出手指姿态。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("ZMOCAP_FRAMES",),
                "tracks": ("ZMOCAP_TRACKS",),
                "models": ("ZMOCAP_MODELS",),
            },
            "optional": {
                "keypoints": ("ZMOCAP_KEYPOINTS",),
                "low_memory": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_HAND_FEATURES",)
    RETURN_NAMES = ("hand_features",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, frames, tracks, models, keypoints=None, low_memory=True):
        assets = build_assets(Path(models["root"]), Path(models["vendor"]))
        count = int(frames["count"])
        boxes = np.asarray(tracks["boxes"], dtype=np.float32)[:count]
        intrinsics = np.asarray(tracks["K"], dtype=np.float32)[:count]
        kp = None if keypoints is None else np.asarray(keypoints["xy"], dtype=np.float32)
        runner = ExMRRunner(assets)
        features = runner.hands(
            Path(frames["dir"]), boxes, intrinsics, keypoints=kp, low_memory=bool(low_memory)
        )
        array = np.asarray(features.detach().cpu() if hasattr(features, "detach") else features)
        print(f"[zmocap] hamer hand features {array.shape}")
        return ({"values": array.astype(np.float32), "count": count},)
