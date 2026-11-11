"""模型加载/自检节点：解析 ComfyUI/models/zmocap 下的权重。"""

from pathlib import Path

from .exmr_engine import SOURCES

REPO = Path(__file__).resolve().parent.parent


def models_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "zmocap"
    except Exception:  # 独立运行时退化到家目录
        return Path.home() / "zext/comfyui/models/zmocap"


class ZMocapModels:
    """选定 zmocap 的模型根目录（默认 ComfyUI/models/zmocap）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models_root": ("STRING", {"default": ""}),
                "low_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ZMOCAP_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, models_root, low_memory):
        root = Path(models_root) if models_root else globals()["models_root"]()
        missing = [rel for rel in SOURCES.values() if not (root / rel).is_file()]
        report = "模型齐全" if not missing else "缺失: " + ", ".join(missing)
        print(f"[zmocap] models={root} low_memory={low_memory} {report}")
        return ({"root": str(root), "vendor": str(REPO / "vendor/ExMR"), "low_memory": low_memory},)
