"""ExMR 解算节点：关键点 + body/hand 特征 → SMPL-X 参数与世界坐标关节。

三个前置阶段各自是独立节点（`ZMocapSapiens2Keypoints` / `ZMocapSam3DBodyFeatures`
/ `ZMocapHaMeRHandFeatures`），这里只做解算 + 前向出关节，便于在 ComfyUI 里
单独缓存、单独观察每一步。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .exmr_engine import ExMRRunner, build_assets


class ZMocapExMR:
    """用预计算的关键点与特征跑 ExMR，输出 SMPL-X 动作与 127 个关节。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("ZMOCAP_KEYPOINTS",),
                "body_features": ("ZMOCAP_BODY_FEATURES",),
                "hand_features": ("ZMOCAP_HAND_FEATURES",),
                "tracks": ("ZMOCAP_TRACKS",),
                "models": ("ZMOCAP_MODELS",),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "static_camera": ("BOOLEAN", {"default": True}),
                "smooth_global": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, keypoints, body_features, hand_features, tracks, models,
            fps=30.0, static_camera=True, smooth_global=True):
        with_hands = hand_features is not None
        if not with_hands:
            print("[zmocap] 未接入手部特征：手部姿态将保持中性")

        assets = build_assets(Path(models["root"]), Path(models["vendor"]))
        count = int(keypoints["count"])
        boxes = np.asarray(tracks["boxes"], dtype=np.float32)[:count]
        intrinsics = np.asarray(tracks["K"], dtype=np.float32)[:count]

        runner = ExMRRunner(assets)
        data = runner.model_inputs(
            np.asarray(keypoints["xy"], dtype=np.float32),
            boxes,
            intrinsics,
            body=np.asarray(body_features["values"], dtype=np.float32),
            hands=np.asarray(hand_features["values"], dtype=np.float32) if with_hands else None,
        )
        native = runner.infer(
            data, static_cam=static_camera, fps=float(fps), smooth_global=smooth_global
        )
        params = {
            key: (value.detach().cpu() if hasattr(value, "detach") else value)
            for key, value in native["smplx_global"].items()
        }
        joints = _joints(params, assets / "body_models/smplx/SMPLX_NEUTRAL.npz")
        motion = {
            "params": params,
            "joints": joints,
            "fps": float(fps),
            "frames": int(count),
            "static_conf_logits": native.get("static_conf_logits"),
            "source": "exmr/sapiens2",
            "has_hands": bool(with_hands),
        }
        _release()
        return (motion,)


def _joints(params, body_path) -> np.ndarray:
    """SMPL-X 前向得到世界坐标关节 (T, 127, 3)，含手指。

    面部的 jaw/leye/reye 必须显式给成 batch 形状：smplx 内部把它们和
    global_orient/body_pose 一起 `torch.cat(dim=1)`，若留默认值（1x3）就会
    在 155 帧时报 "Expected size 155 but got size 1 for tensor number 2"。
    """
    import torch
    import smplx

    body = smplx.SMPLX(str(body_path), flat_hand_mean=False, use_pca=False).eval()
    count = int(params["betas"].shape[0])

    def get(name, width, default=0.0):
        value = params.get(name)
        if value is None:
            return torch.full((count, width), default)
        return value.clone().reshape(count, width)

    with torch.inference_mode():
        output = body(
            betas=get("betas", 10),
            global_orient=get("global_orient", 3),
            body_pose=get("body_pose", 63),
            left_hand_pose=get("left_hand_pose", 45),
            right_hand_pose=get("right_hand_pose", 45),
            expression=get("expression", 10),
            jaw_pose=get("jaw_pose", 3),
            leye_pose=get("leye_pose", 3),
            reye_pose=get("reye_pose", 3),
            transl=get("transl", 3),
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
