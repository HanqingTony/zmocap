"""节点注册表。"""

from .exmr import ZMocapExMR
from .export import ZMocapExport
from .loader import ZMocapModels
from .preview import ZMocapPreview3D
from .probe import ZMocapEnvironment
from .track import ZMocapTrack

NODE_CLASS_MAPPINGS = {
    "ZMocapEnvironment": ZMocapEnvironment,
    "ZMocapModels": ZMocapModels,
    "ZMocapTrack": ZMocapTrack,
    "ZMocapExMR": ZMocapExMR,
    "ZMocapPreview3D": ZMocapPreview3D,
    "ZMocapExport": ZMocapExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZMocapEnvironment": "zmocap · 环境自检",
    "ZMocapModels": "zmocap · 模型",
    "ZMocapTrack": "zmocap · 人物跟踪",
    "ZMocapExMR": "zmocap · ExMR 骨骼",
    "ZMocapPreview3D": "zmocap · 3D 预览",
    "ZMocapExport": "zmocap · 导出",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
