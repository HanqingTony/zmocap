"""节点注册表。

流水线（每个阶段一个节点，便于在 ComfyUI 里观察/缓存）::

    LoadVideo → Video Slice → VideoFrameSample → GetVideoComponents   (ComfyUI 官方)
      → ZMocapTrack → ZMocapFrames
      → ZMocapSapiens2Keypoints → ZMocapSam3DBodyFeatures → ZMocapHaMeRHandFeatures
      → ZMocapExMR → ZMocapFootContact → ZMocapSMPLPreviewer
                    │                 └→ ZMocapExport（NPZ/BVH + contacts JSON/NPZ）
                    └→ SMPL2SOMA → SOMA2VRM → ZMocapVRMPreviewer ← ZMocapVRMLoader

角色加载与绑定预览彻底解耦：VRM Loader 只负责“载入角色”，Rig 只负责“动作 + 角色 → GLB”。
"""

from .body_features import ZMocapSam3DBodyFeatures
from .character import ZMocapVRMLoader
from .exmr import ZMocapExMR
from .foot_contact import ZMocapContactLoad, ZMocapFootContact
from .export import ZMocapExport
from .format_soma import ZMocapSMPL2SOMA, ZMocapSOMA2VRM
from .frames_node import ZMocapFrames, ZMocapIntrinsics
from .hands import ZMocapHaMeRHandFeatures
from .keypoints import ZMocapSapiens2Keypoints
from .loader import ZMocapModels
from .video import ZMocapLoadVideo
from .probe import ZMocapEnvironment
from .previewer import ZMocapSMPLPreviewer, ZMocapVRMPreviewer
from .track import ZMocapTrack

NODE_CLASS_MAPPINGS = {
    "ZMocapEnvironment": ZMocapEnvironment,
    "ZMocapLoadVideo": ZMocapLoadVideo,
    "ZMocapModels": ZMocapModels,
    "ZMocapTrack": ZMocapTrack,
    "ZMocapFrames": ZMocapFrames,
    "ZMocapIntrinsics": ZMocapIntrinsics,
    "ZMocapSapiens2Keypoints": ZMocapSapiens2Keypoints,
    "ZMocapSam3DBodyFeatures": ZMocapSam3DBodyFeatures,
    "ZMocapHaMeRHandFeatures": ZMocapHaMeRHandFeatures,
    "ZMocapExMR": ZMocapExMR,
    "ZMocapFootContact": ZMocapFootContact,
    "ZMocapContactLoad": ZMocapContactLoad,
    "ZMocapSMPLPreviewer": ZMocapSMPLPreviewer,
    "ZMocapVRMPreviewer": ZMocapVRMPreviewer,
    "ZMocapSMPL2SOMA": ZMocapSMPL2SOMA,
    "ZMocapSOMA2VRM": ZMocapSOMA2VRM,
    "ZMocapVRMLoader": ZMocapVRMLoader,
    "ZMocapExport": ZMocapExport,
}

# 仅 ASCII，不带序号
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZMocapEnvironment": "zmocap Environment",
    "ZMocapLoadVideo": "zmocap Load Video",
    "ZMocapModels": "zmocap Models",
    "ZMocapTrack": "zmocap Track",
    "ZMocapFrames": "zmocap Frames",
    "ZMocapIntrinsics": "zmocap Intrinsics",
    "ZMocapSapiens2Keypoints": "zmocap Sapiens2 Keypoints",
    "ZMocapSam3DBodyFeatures": "zmocap SAM3D Body Features",
    "ZMocapHaMeRHandFeatures": "zmocap HaMeR Hand Features",
    "ZMocapExMR": "zmocap ExMR Solve",
    "ZMocapFootContact": "zmocap Foot Contact",
    "ZMocapContactLoad": "zmocap Contact Load",
    "ZMocapSMPLPreviewer": "zmocap SMPL Previewer",
    "ZMocapVRMPreviewer": "zmocap VRM Previewer",
    "ZMocapSMPL2SOMA": "zmocap SMPL2SOMA",
    "ZMocapSOMA2VRM": "zmocap SOMA2VRM",
    "ZMocapVRMLoader": "zmocap VRM Loader",
    "ZMocapExport": "zmocap Export NPZ/BVH",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
