"""节点注册表。

流水线（每个阶段一个节点，便于在 ComfyUI 里观察/缓存）::

    LoadVideo → Video Slice → VideoFrameSample → GetVideoComponents   (ComfyUI 官方)
      → ZMocapTrack → ZMocapFrames
      → ZMocapSapiens2Keypoints → ZMocapSam3DBodyFeatures → ZMocapHaMeRHandFeatures
      → ZMocapExMR → ZMocapPreview3D / ZMocapExport
"""

from .body_features import ZMocapSam3DBodyFeatures
from .character import ZMocapCharacterPreview
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
    "ZMocapCharacterPreview": ZMocapCharacterPreview,
    "ZMocapPreview3D": ZMocapPreview3D,
    "ZMocapExport": ZMocapExport,
}

# 仅 ASCII，不带序号
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZMocapEnvironment": "zmocap Environment",
    "ZMocapModels": "zmocap Models",
    "ZMocapTrack": "zmocap Track",
    "ZMocapFrames": "zmocap Frames",
    "ZMocapIntrinsics": "zmocap Intrinsics",
    "ZMocapSapiens2Keypoints": "zmocap Sapiens2 Keypoints",
    "ZMocapSam3DBodyFeatures": "zmocap SAM3D Body Features",
    "ZMocapHaMeRHandFeatures": "zmocap HaMeR Hand Features",
    "ZMocapExMR": "zmocap ExMR Solve",
    "ZMocapRig": "zmocap Rig (GLB)",
    "ZMocapCharacterPreview": "zmocap Character Preview",
    "ZMocapPreview3D": "zmocap Skeleton Preview",
    "ZMocapExport": "zmocap Export NPZ/BVH",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
