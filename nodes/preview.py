"""3D 预览节点：把关节序列推给前端 JS 在节点内渲染。"""

import numpy as np


class ZMocapPreview3D:
    """在节点内部播放骨骼动画（前端 canvas 投影渲染，无需 ComfyUI 时间轴）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"motion": ("ZMOCAP_MOTION",)},
            "optional": {"max_frames": ("INT", {"default": 600, "min": 1, "max": 5000})},
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "zmocap"
    OUTPUT_NODE = True

    def run(self, motion, max_frames=600):
        joints = np.asarray(motion["joints"], dtype=np.float32)[: int(max_frames)]
        return {
            "ui": {
                "zmocap_joints": [joints.tolist()],
                "zmocap_fps": [float(motion.get("fps", 30.0))],
            }
        }
