"""ExMR 骨骼节点：帧批 → SMPL-X 参数与世界坐标关节。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from .exmr_engine import ExMRRunner, build_assets
from .frames import images_to_dir


class ZMocapExMR:
    """用 Sapiens2 关键点 + SAM 3D Body/HaMeR 特征，跑 ExMR 得到 SMPL-X 动作。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tracks": ("ZMOCAP_TRACKS",),
                "models": ("ZMOCAP_MODELS",),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "static_camera": ("BOOLEAN", {"default": True}),
                "smooth_global": ("BOOLEAN", {"default": True}),
                "keep_frames": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, images, tracks, models, fps=30.0, static_camera=True, smooth_global=True, keep_frames=False):
        import torch
        from exmr import pipeline as P

        assets = build_assets(Path(models["root"]), Path(models["vendor"]))
        low_memory = bool(models.get("low_memory", True))

        workdir = Path(tempfile.mkdtemp(prefix="zmocap-frames-"))
        try:
            frames_dir, count, _ = images_to_dir(images, workdir)
            boxes = np.asarray(tracks["boxes"], dtype=np.float32)[:count]
            intrinsics = np.asarray(tracks["K"], dtype=np.float32)[:count]
            if boxes.shape[0] != count:
                raise ValueError(f"跟踪帧数 {boxes.shape[0]} 与视频帧数 {count} 不一致")

            runner = ExMRRunner(assets)
            keypoints = runner.keypoints(frames_dir, boxes, low_memory=low_memory)
            body, hands = runner.features(
                frames_dir, boxes, intrinsics, keypoints=keypoints, low_memory=low_memory
            )
            data = runner.model_inputs(keypoints, boxes, intrinsics, body=body, hands=hands)
            native = runner.infer(
                data, static_cam=static_camera, fps=float(fps), smooth_global=smooth_global
            )
            params = {k: v.detach().cpu() if hasattr(v, "detach") else torch.as_tensor(v) for k, v in native["smplx_global"].items()}
            joints = _joints(params, assets / "body_models/smplx/SMPLX_NEUTRAL.npz")
            motion = {
                "params": params,
                "joints": joints,
                "fps": float(fps),
                "frames": int(count),
                "static_conf_logits": native.get("static_conf_logits"),
                "source": "exmr/sapiens2",
            }
            _release()
            return (motion,)
        finally:
            if not keep_frames:
                shutil.rmtree(workdir, ignore_errors=True)


def _joints(params, body_path) -> np.ndarray:
    """SMPL-X 前向得到世界坐标关节 (T, J, 3)。"""
    import torch
    import smplx

    body = smplx.SMPLX(str(body_path), flat_hand_mean=False, use_pca=False).eval()
    count = int(params["betas"].shape[0])
    with torch.inference_mode():
        output = body(
            betas=params["betas"].clone(),
            global_orient=params["global_orient"].clone(),
            body_pose=params["body_pose"].clone(),
            left_hand_pose=params.get("left_hand_pose", torch.zeros(count, 45)),
            right_hand_pose=params.get("right_hand_pose", torch.zeros(count, 45)),
            transl=params["transl"].clone(),
        )
    return output.joints.detach().cpu().numpy()


def _release():
    import gc

    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        pass


import torch as _torch  # noqa: E402  (供 _joints 内默认张量使用)
_ = _torch
