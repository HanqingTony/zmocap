"""模型加载/自检节点：解析 ComfyUI/models/zmocap 下的权重。"""

from pathlib import Path

from .exmr_engine import SOURCES

REPO = Path(__file__).resolve().parent.parent


# 权重齐全的判定标志（ExMR 主模型）
MARKER = "exmr/exmr.safetensors"


def candidate_roots():
    """候选模型根目录：ComfyUI 自己的 models、extra 路径、家目录备份。"""
    roots = []
    try:
        import folder_paths

        roots.append(Path(folder_paths.models_dir) / "zmocap")
        base = getattr(folder_paths, "base_path", None)
        if base:
            roots.append(Path(base) / "models" / "zmocap")
    except Exception:
        pass
    roots.append(Path.home() / "zext/comfyui/models/zmocap")
    return roots


def models_root() -> Path:
    """挑第一个真正有 ExMR 权重的目录（都不全就返回首选目录）。"""
    roots = candidate_roots()
    for root in roots:
        if (root / MARKER).is_file():
            return root
    return roots[0]


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
