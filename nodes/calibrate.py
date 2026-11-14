"""绑定拆成两段：**标定**（角色 → 静态对齐参数）与**求解**（动作 + 标定 → GLB）。

这样做的目的：把"角色为什么绑得不对"变成可观察的中间结果。
以前预览器里既读角色、又做标定、又求解、又写文件，任何一步偏了都只能看最终画面猜；
现在中间有 ``ZMOCAP_CALIBRATION`` 可以单独检查（骨长比、未匹配骨骼、髋骨命中）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _output_dir() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory())
    except Exception:
        root = Path("output")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _character_path(character) -> str:
    target = str(character.get("path") or "") if isinstance(character, dict) else str(character or "")
    target = target.strip()
    return "" if target.startswith("(") else target


class ZMocapCalibrate:
    """角色 → 标定结果（与动作无关，可复用）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": ("ZMOCAP_CHARACTER",),
            }
        }

    RETURN_TYPES = ("ZMOCAP_CALIBRATION", "STRING")
    RETURN_NAMES = ("calibration", "report")
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "读取角色骨架，算出骨长比与每根骨的 rest 对齐矩阵；不改动角色文件。"

    def run(self, character):
        from ..soma.binding import calibrate

        target = _character_path(character)
        if not target:
            raise ValueError("Calibrate 需要一个角色（VRM/GLB），不能用内置骨架")
        calibration = calibrate(target)
        report = dict(calibration["report"])
        report["character"] = Path(target).name
        report["hips_bone"] = calibration.get("hips_bone")
        print(f"[zmocap] Calibrate {report['character']}: {report['bones']} 骨, scale {report['scale']}")
        return (calibration, json.dumps(report, ensure_ascii=True))


class ZMocapBind:
    """动作 + 标定 → 带动画 GLB（写到 output，供预览/导出）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_VRM",),
                "calibration": ("ZMOCAP_CALIBRATION",),
            },
            "optional": {
                "filename": ("STRING", {"default": "zmocap_bound"}),
                "sticks": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_GLB", "STRING")
    RETURN_NAMES = ("glb", "report")
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "用标定把动作套到角色骨架上并写出 GLB（这一步才真正改骨骼动画）。"

    def run(self, motion, calibration, filename="zmocap_bound", sticks=True):
        from ..motion import validate_clip
        from ..soma.binding import apply_calibration, write_bound_glb

        validate_clip(motion, "vrm")
        target = str(calibration.get("character") or "")
        if not target or not Path(target).is_file():
            raise ValueError(f"标定结果里的角色文件不存在：{target!r}，请重新运行 Calibrate")

        solved = apply_calibration(motion, calibration)
        digest = hashlib.sha1(
            "|".join([
                str(motion.get("digest")), str(calibration.get("scale")), target, str(sticks),
            ]).encode()
        ).hexdigest()[:8]
        outdir = _output_dir()
        out_path = outdir / f"{filename}_{digest}.glb"
        path, report = write_bound_glb(solved, target, out_path, sticks=sticks)
        relative = Path(path).relative_to(outdir).as_posix()
        glb = {
            "format": "zmocap.glb/1",
            "path": str(path),
            "relative": relative,
            "report": report,
            "motion_digest": motion.get("digest"),
            "scale": float(calibration.get("scale", 1.0)),
        }
        print(f"[zmocap] Bind {Path(target).name}: {report['bones']} 骨, scale {glb['scale']:.4f} → {relative}")
        return (glb, json.dumps({**report, "glb": relative}, ensure_ascii=True))
