"""节点注册表。

流水线（每个阶段一个节点，便于在 ComfyUI 里观察/缓存）::

    LoadVideo → Video Slice → VideoFrameSample → GetVideoComponents   (ComfyUI 官方)
      → ZMocapTrack → ZMocapFrames
      → ZMocapSapiens2Keypoints → ZMocapSam3DBodyFeatures → ZMocapHaMeRHandFeatures
      → ZMocapExMR → ZMocapPreview3D / ZMocapExport
"""

from .body_features import ZMocapSam3DBodyFeatures
from .exmr import ZMocapExMR
from .export import ZMocapExport
from .frames_node import ZMocapFrames, ZMocapIntrinsics
from .hands import ZMocapHaMeRHandFeatures
from .keypoints import ZMocapSapiens2Keypoints
from .loader import ZMocapModels
from .preview import ZMocapPreview3D
from .probe import ZMocapEnvironment
from .retarget import ZMocapRig
from .track import ZMocapTrack

NODE_CLASS_MAPPINGS = {
    "ZMocapEnvironment": ZMocapEnvironment,
    "ZMocapModels": ZMocapModels,
    "ZMocapTrack": ZMocapTrack,
    "ZMocapFrames": ZMocapFrames,
    "ZMocapIntrinsics": ZMocapIntrinsics,
    "ZMocapSapiens2Keypoints": ZMocapSapiens2Keypoints,
    "ZMocapSam3DBodyFeatures": ZMocapSam3DBodyFeatures,
    "ZMocapHaMeRHandFeatures": ZMocapHaMeRHandFeatures,
    "ZMocapExMR": ZMocapExMR,
    "ZMocapRig": ZMocapRig,
    "ZMocapPreview3D": ZMocapPreview3D,
    "ZMocapExport": ZMocapExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZMocapEnvironment": "zmocap · 环境自检",
    "ZMocapModels": "zmocap · 模型",
    "ZMocapTrack": "zmocap · ① 人物跟踪",
    "ZMocapFrames": "zmocap · 帧落盘",
    "ZMocapIntrinsics": "zmocap · 相机内参",
    "ZMocapSapiens2Keypoints": "zmocap · ② Sapiens2 关键点",
    "ZMocapSam3DBodyFeatures": "zmocap · ③ SAM 3D Body 特征",
    "ZMocapHaMeRHandFeatures": "zmocap · ④ HaMeR 手部特征",
    "ZMocapExMR": "zmocap · ⑤ ExMR 解算",
    "ZMocapRig": "zmocap · ⑥ 角色绑定预览 (GLB)",
    "ZMocapPreview3D": "zmocap · 骨骼预览",
    "ZMocapExport": "zmocap · 导出 NPZ/BVH",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
