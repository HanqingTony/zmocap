"""视频帧与张量之间的转换工具（IMAGE ⇄ 磁盘帧目录）。"""

from pathlib import Path

import numpy as np


def images_to_dir(images, directory, prefix="frame", ext="png"):
    """把 ComfyUI 的 IMAGE 批张量写成 ExMR 期望的帧目录。

    images: (B, H, W, C) float [0, 1] 的 torch 张量（ComfyUI 约定）
    返回: 帧目录路径、帧数、画布尺寸 (W, H)
    """
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    if array.ndim == 3:
        array = array[None]
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0 + 0.5).astype(np.uint8)
    for index, frame in enumerate(array):
        Image.fromarray(frame).save(directory / f"{prefix}{index:06d}.{ext}")
    height, width = array.shape[1], array.shape[2]
    return directory, array.shape[0], (width, height)


def default_intrinsics(width, height, fov_deg=55.0):
    """用水平视场角假设生成内参矩阵 (3, 3)。缺少相机标定时的合理默认值。"""
    focal = 0.5 * width / np.tan(np.radians(fov_deg) * 0.5)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
