"""两个预览器：形态一致（motion + 可选角色 + skinned 开关），只接受不同的 motion。

    ZMocap SMPL Previewer   motion(smplx) → GLB（SMPL-X 自带骨架 / 绑角色）
    ZMocap VRM Previewer    motion(vrm)   → GLB（标准 humanoid / 绑角色）

两者的 `skinned` 只决定前端**初始显示层**：开 = 蒙皮网格，关 = 骨骼小棍；
两套几何都写在同一份 GLB 里，切换不需要重跑。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None


def _output_dir() -> Path:
    return Path(folder_paths.get_output_directory()) if folder_paths else Path("/tmp")


def _character_path(character) -> str:
    target = str(character.get("path") or "") if isinstance(character, dict) else str(character or "")
    target = target.strip()
    return "" if target.startswith("(") else target


def _digest(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


class ZMocapSMPLPreviewer:
    """SMPL-X 运动的预览器（自带骨架或绑到角色）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_MOTION",),
                "filename": ("STRING", {"default": "zmocap_smpl"}),
            },
            "optional": {
                "character": ("ZMOCAP_CHARACTER",),
                "skinned": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "预览 SMPL-X 运动：蒙皮视图与骨骼小棍视图写在同一个 GLB，切换即时。"

    def run(self, motion, filename, character=None, skinned=True):
        from .retarget import build_glb
        from .smpl_mesh import write_smpl_glb

        skeleton = str(motion.get("skeleton") or ("smplx" if "params" in motion else "unknown"))
        if skeleton != "smplx":
            raise ValueError(f"SMPL Previewer 需要 SMPL-X 运动，收到 {skeleton}")
        target = _character_path(character)

        if not target:
            # 自带 SMPL-X 网格：参数与网格同源，零重定向，用作对照基准
            outdir = _output_dir()
            frames = int(np.asarray(motion["params"]["global_orient"]).shape[0])
            digest = _digest(["smplx", str(frames), f"{float(motion.get('fps', 30.0)):.3f}"])
            path, info = write_smpl_glb(motion, outdir / f"{filename}_{digest}.glb")
            relative = path.relative_to(outdir).as_posix()
            return {
                "ui": {"zmocap_glb": [relative],
                       "zmocap_layer": ["skinned" if skinned else "bones"],
                       "zmocap_rig": [{"path": str(path), **info}]},
                "result": (relative,),
            }

        joints = np.asarray(motion["joints"])
        digest = _digest([
            "smpl", str(joints.shape[0]), f"{float(motion.get('fps', 30.0)):.3f}",
            f"{float(np.abs(joints).sum()):.4f}", target,
        ])
        outdir = _output_dir()
        path, info = build_glb(motion, outdir / f"{filename}_{digest}.glb",
                               character=target, mode="auto" if target else "mannequin",
                               strict=bool(target), bones=True)
        relative = path.relative_to(outdir).as_posix()
        print(f"[zmocap] SMPL preview {info['mode']} -> {path.name} ({info['bones']} bones)")
        return {
            "ui": {"zmocap_glb": [relative], "zmocap_layer": ["skinned" if skinned else "bones"],
                   "zmocap_rig": [{"path": str(path), **info}]},
            "result": (relative,),
        }


class ZMocapVRMPreviewer:
    """标准 humanoid 运动（skeleton='vrm'）的预览器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_MOTION",),
                "filename": ("STRING", {"default": "zmocap_vrm"}),
            },
            "optional": {
                "character": ("ZMOCAP_CHARACTER",),
                "skinned": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "预览 humanoid 运动：绑角色出蒙皮 + 骨骼小棍，切换即时，不重跑。"

    def run(self, motion, filename, character=None, skinned=True):
        from ..soma.binding import write_character_glb, write_humanoid_glb

        skeleton = str(motion.get("skeleton") or "")
        if skeleton != "vrm":
            raise ValueError(f"VRM Previewer 需要 skeleton='vrm' 的运动，收到 {skeleton or '未知'}")
        target = _character_path(character)
        animation = np.asarray(motion["world_rotation"])
        digest = _digest([
            "vrm", str(animation.shape[0]), f"{float(motion.get('fps', 30.0)):.3f}",
            f"{float(np.abs(animation).sum()):.6f}",
            f"{float(np.abs(np.asarray(motion['transl'])).sum()):.6f}", target,
        ])
        outdir = _output_dir()
        out_path = outdir / f"{filename}_{digest}.glb"
        if target:
            path, report = write_character_glb(motion, target, out_path)
        else:
            path, report = write_humanoid_glb(motion, out_path)
        relative = Path(path).relative_to(outdir).as_posix()
        print(f"[zmocap] VRM preview {Path(path).name} ({report['bones']} bones, scale {report.get('scale', 1.0)})")
        return {
            "ui": {"zmocap_glb": [relative], "zmocap_layer": ["skinned" if skinned else "bones"],
                   "zmocap_rig": [{"path": str(path), **report}]},
            "result": (relative,),
        }
