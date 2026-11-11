"""节点注册表。"""

from .probe import ZMocapEnvironment

NODE_CLASS_MAPPINGS = {
    "ZMocapEnvironment": ZMocapEnvironment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZMocapEnvironment": "zmocap · 环境自检",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
