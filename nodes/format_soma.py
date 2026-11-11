"""SOMA 枢纽节点：SMPL-X → SOMA → 目标格式。

重定向不再"把源增量叠到角色静止姿态"，而是走 SOMA 通用骨架并用方向+rest 求解，
所以源与角色静止姿态不同（T-pose / A-pose）、朝向差 180° 也能正确对齐。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

try:
    import folder_paths
except Exception:  # pragma: no cover - 便于离线单测
    folder_paths = None


OUTPUT_DIR = Path(folder_paths.get_output_directory()) if folder_paths else Path("/tmp")


def _character_options():
    names = ["(内置骨架)"]
    root = Path(folder_paths.get_folder_paths("zmocap_characters")[0]) if folder_paths and folder_paths.get_folder_paths("zmocap_characters") else None
    if root and root.is_dir():
        names += sorted(p.name for p in root.iterdir() if p.suffix.lower() in {".vrm", ".glb"})
    return names


class ZMocapSMPL2SOMA:
    """ExMR 的 SMPL-X 参数 → 78 关节 SOMA 运动（通用动作交换格式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_MOTION",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.5}),
            },
            "optional": {
                "save_npz": ("BOOLEAN", {"default": True, "label_on": "存到 output", "label_off": "只在内存"}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_SOMA", "STRING")
    RETURN_NAMES = ("soma", "info")
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "把 SMPL-X 参数转成 SOMA(78 关节) 通用运动，作为跨软件交换的中间格式。"

    def run(self, motion, fps=30.0, save_npz=True):
        from ..soma.motion import build_soma_motion

        params = motion.get("params") or motion if isinstance(motion, dict) else motion
        if isinstance(params, dict) and "smplx_global" in params:
            params = params["smplx_global"]
        if not isinstance(params, dict) or "body_pose" not in params:
            raise ValueError("需要 ZMocapExMR 输出的 motion（内含 SMPL-X 参数）")
        motion = build_soma_motion(params, fps=float(fps))
        detail = {
            "joints": len(motion.names),
            "frames": motion.frames,
            "fps": motion.fps,
            "height_m": round(float(motion.positions[:, motion.index("Head"), 1].max()), 3),
        }
        if save_npz:
            path = OUTPUT_DIR / f"zmocap_soma_{uuid.uuid4().hex[:8]}.npz"
            motion.save(path)
            detail["npz"] = path.name
        print(f"[zmocap] SMPL2SOMA {detail['frames']} 帧 → {detail['joints']} 关节")
        return (motion, json.dumps(detail, ensure_ascii=True))


class ZMocapSOMA2VRM:
    """SOMA 运动 → 指定角色（VRM/GLB）的带动画 GLB。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soma": ("ZMOCAP_SOMA",),
                "character": ("ZMOCAP_CHARACTER",),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "keep_on_ground": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb", "info")
    FUNCTION = "run"
    CATEGORY = "zmocap"
    OUTPUT_NODE = True
    DESCRIPTION = "用 SOMA 通用骨架把动作重定向到任意角色，输出带动画的 GLB（可直接接 Preview3D）。"

    def run(self, soma, character, fps=30.0, keep_on_ground=True):
        from ..soma.vrm import retarget_to_rig, write_animation
        from .character import resolve_character

        source = resolve_character(character)
        retargeted = retarget_to_rig(soma, source)
        path = OUTPUT_DIR / f"zmocap_character_{uuid.uuid4().hex[:8]}.glb"
        write_animation(source, path, retargeted, fps)

        mapping = retargeted["rigmap"]
        detail = {
            "character": Path(source).name,
            "bones_animated": len(retargeted["world_rotation"]),
            "frames": int(retargeted["frames"]),
            "unmapped": mapping.missing(),
            "notes": list(mapping.notes)[:4],
        }
        print(f"[zmocap] SOMA2VRM {Path(source).name}: {detail['bones_animated']} 骨骼, {detail['frames']} 帧")
        return {"ui": {"zmocap_glb": [path.name]}, "result": (path.name, json.dumps(detail, ensure_ascii=True))}
