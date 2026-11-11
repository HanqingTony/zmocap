"""纯格式转换节点：单进单出，只处理 motion，不碰角色文件。

    LM:MOTION(smplx) --SMPL2SOMA--> LM:MOTION(soma) --SOMA2VRM--> LM:MOTION(vrm)

角色只在最后绑定/导出时出现（`ZMocap Rig` 吃 motion + 角色）。
"""

from __future__ import annotations

import numpy as np

MOTION = "ZMOCAP_MOTION"


def _skeleton(motion: dict) -> str:
    if not isinstance(motion, dict):
        raise TypeError("输入不是 motion")
    return str(motion.get("skeleton") or ("smplx" if "params" in motion else "unknown"))


class ZMocapSMPL2SOMA:
    """SMPL-X 运动 → SOMA(78 关节) 运动。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"motion": (MOTION,)}}

    RETURN_TYPES = (MOTION,)
    RETURN_NAMES = ("motion",)
    OUTPUT_TOOLTIPS = ("SOMA 骨架上的运动（skeleton='soma'），可直接接 SOMA2VRM",)
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "把 SMPL-X 运动重新表示为 SOMA(78 关节) 通用运动，作为跨骨架交换的中间格式。"

    def run(self, motion):
        from ..soma.motion import build_soma_motion

        skeleton = _skeleton(motion)
        if skeleton != "smplx":
            raise ValueError(f"SMPL2SOMA 需要 SMPL-X 运动，收到 {skeleton}")
        result = build_soma_motion(motion["params"], fps=float(motion.get("fps", 30.0)))
        out = {
            "skeleton": "soma",
            "fps": result.fps,
            "frames": result.frames,
            "names": list(result.names),
            "parents": np.asarray(result.parents),
            "rest": np.asarray(result.rest),
            "world": np.asarray(result.world),
            "source": "smpl2soma",
        }
        print(f"[zmocap] SMPL2SOMA: {out['frames']} 帧 × {len(out['names'])} 关节")
        return (out,)


class ZMocapSOMA2VRM:
    """SOMA 运动 → 标准 VRM humanoid 骨架上的运动。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"motion": (MOTION,)}}

    RETURN_TYPES = (MOTION,)
    RETURN_NAMES = ("motion",)
    OUTPUT_TOOLTIPS = ("规范 humanoid 骨架上的运动（skeleton='vrm'），接 Rig 绑定角色",)
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "把 SOMA 运动翻译到 VRM 标准人形骨架（55 骨，含五指），与具体角色无关。"

    def run(self, motion):
        from ..soma.motion import SomaMotion
        from ..soma.vrm_motion import soma_to_vrm

        skeleton = _skeleton(motion)
        if skeleton != "soma":
            raise ValueError(f"SOMA2VRM 需要 SOMA 运动，收到 {skeleton}")
        soma = SomaMotion(
            names=list(motion["names"]),
            parents=np.asarray(motion["parents"]),
            rest=np.asarray(motion["rest"]),
            world=np.asarray(motion["world"]),
            fps=float(motion["fps"]),
        )
        out = soma_to_vrm(soma)
        out["source"] = "soma2vrm"
        print(f"[zmocap] SOMA2VRM: {out['frames']} 帧 × {len(out['names'])} 骨")
        return (out,)
