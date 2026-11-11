"""Sapiens2 关键点节点（阶段 1）：帧目录 + 跟踪框 → 每帧 63 个身体关键点。

对应 vendored `exmr.preprocessing.keypoints.sapiens2.Sapiens2Keypoints`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .exmr_engine import ExMRRunner, build_assets


class ZMocapSapiens2Keypoints:
    """用 Sapiens2 1B Pose 从每帧的人体裁剪里解码 63 个关键点。"""

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

    RETURN_TYPES = ("ZMOCAP_KEYPOINTS",)
    RETURN_NAMES = ("keypoints",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, frames, tracks, models, low_memory=True):
        _reserve(1.6)
        assets = build_assets(Path(models["root"]), Path(models["vendor"]))
        count = int(frames["count"])
        boxes = np.asarray(tracks["boxes"], dtype=np.float32)[:count]
        runner = ExMRRunner(assets)
        keypoints = runner.keypoints(Path(frames["dir"]), boxes, low_memory=bool(low_memory))
        array = np.asarray(keypoints.detach().cpu() if hasattr(keypoints, "detach") else keypoints)
        print(f"[zmocap] sapiens2 keypoints {array.shape}")
        return ({"xy": array.astype(np.float32), "count": count,
                 "frames": frames["dir"], "backend": "sapiens2"},)


def _reserve(gib: float):
    try:
        import comfy.model_management as mm

        mm.free_memory(int(gib * 1024 ** 3), mm.get_torch_device())
    except Exception:
        pass
