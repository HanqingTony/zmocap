"""环境自检节点：确认 zmocap 能跑在 ComfyUI 自己的 Python 环境里。"""

import platform

import torch


class ZMocapEnvironment:
    """打印运行时环境（python / torch / CUDA / 设备 / 模型目录）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self):
        lines = [
            f"python {platform.python_version()} ({platform.system()})",
            f"torch {torch.__version__} (cuda {torch.version.cuda})",
            f"cuda available: {torch.cuda.is_available()}",
        ]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                lines.append(
                    f"gpu{i}: {torch.cuda.get_device_name(i)} "
                    f"{total / 2**30:.1f} GiB (free {free / 2**30:.1f} GiB)"
                )
        try:
            import folder_paths

            lines.append(f"models: {folder_paths.models_dir}")
        except Exception as exc:  # 不在 ComfyUI 里跑时允许降级
            lines.append(f"models: unavailable ({exc})")
        try:
            import comfy.model_management as mm

            lines.append(f"comfy device: {mm.get_torch_device()} / offload {mm.unet_offload_device()}")
        except Exception as exc:
            lines.append(f"comfy model_management: unavailable ({exc})")
        return ("\n".join(lines),)
