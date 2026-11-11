"""帧节点：把 IMAGE 批落盘成 vendored 推理管线要读的帧目录，并向下游传递路径。

ExMR 的预处理按目录读帧（`%06d.jpg`），所以帧目录是这一族节点的公共载体：
`sapiens2 keypoints` / `sam3d body features` / `hamer hand features` 都只需要
它 + 跟踪框，互不依赖彼此的中间结果。
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .frames import default_intrinsics, images_to_dir

# 帧目录在磁盘上的存活时间：ComfyUI 一次执行内的多个下游节点共享它。
_KEEP = {}


class ZMocapFrames:
    """IMAGE 批 → 帧目录（`000000.jpg`），输出可复用的帧包。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {"quality": ("INT", {"default": 95, "min": 60, "max": 100, "step": 1})},
        }

    RETURN_TYPES = ("ZMOCAP_FRAMES",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, images, quality=95):
        root = Path(tempfile.mkdtemp(prefix="zmocap-frames-"))
        directory, count, (width, height) = images_to_dir(images, root, quality=quality)
        bundle = {
            "dir": str(directory),
            "root": str(root),
            "count": int(count),
            "width": int(width),
            "height": int(height),
            "token": uuid.uuid4().hex[:8],
        }
        _KEEP[bundle["token"]] = bundle["root"]
        print(f"[zmocap] frames {count} @ {width}x{height} -> {directory}")
        return (bundle,)


class ZMocapIntrinsics:
    """在缺少相机标定时生成逐帧内参（水平 FOV 假设），也可直接喂已有内参。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("ZMOCAP_TRACKS",),
                "fov_deg": ("FLOAT", {"default": 55.0, "min": 5.0, "max": 150.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("ZMOCAP_TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, tracks, fov_deg=55.0):
        count = int(np.asarray(tracks["boxes"]).shape[0])
        width, height = tracks.get("width"), tracks.get("height")
        if width is None or height is None:
            raise ValueError("tracks 缺少画布尺寸，无法生成内参")
        k = default_intrinsics(int(width), int(height), float(fov_deg))
        return ({"boxes": tracks["boxes"], "K": np.repeat(k[None], count, axis=0),
                 "width": int(width), "height": int(height)},)


def cleanup(token: str) -> None:
    """释放帧目录（导出/预览等收尾节点可调用）。"""
    root = _KEEP.pop(token, None)
    if root:
        shutil.rmtree(root, ignore_errors=True)
