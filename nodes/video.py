"""视频加载节点：把视频文件解码成 ComfyUI 的 IMAGE 批张量。

自带解码（PyAV / ffmpeg 同源），不依赖第三方视频节点包。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg")


class ZMocapLoadVideo:
    """读取一个视频文件为 IMAGE 批（RGB，[0,1]），并给出实际采样帧率与帧数。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "", "multiline": False}),
                "fps": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5},
                ),
                "max_frames": ("INT", {"default": 300, "min": 1, "max": 5000, "step": 1}),
            },
            "optional": {
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("images", "fps", "frames")
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, video, fps=0.0, max_frames=300, start_frame=0):
        import av
        import torch

        path = Path(str(video)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"视频不存在：{path}")
        if path.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"不支持的视频格式：{path.suffix}")

        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise ValueError(f"没有视频轨：{path}")
            source_fps = float(stream.average_rate) if stream.average_rate else 30.0
            target_fps = float(fps) if fps and fps > 0 else source_fps
            step = max(1, round(source_fps / target_fps)) if target_fps < source_fps else 1
            effective_fps = source_fps / step

            frames = []
            for index, frame in enumerate(container.decode(stream)):
                if index < int(start_frame) or (index - int(start_frame)) % step:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= int(max_frames):
                    break

        if not frames:
            raise ValueError(f"没有解出帧：{path}（检查 start_frame/max_frames）")

        array = np.stack(frames).astype(np.float32) / 255.0
        print(f"[zmocap] {path.name}: {len(frames)} 帧 @ {effective_fps:.2f} fps, {array.shape[2]}x{array.shape[1]}")
        return (torch.from_numpy(array), float(effective_fps), int(len(frames)))
