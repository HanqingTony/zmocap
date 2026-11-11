"""角色素材节点：扫描 ``models/zmocap/characters`` 里的 GLB/VRM 并提供预览。

用途：
- ``ZMocapCharacterPreview``：把角色文件原样交给官方 ``Preview3D`` 看外观
  （这样挑角色时不用先跑推理）。
- ``character_choices()``：给 ``ZMocapRig`` 的下拉框用，避免手输路径。
"""

from __future__ import annotations

import shutil
from pathlib import Path

MANNEQUIN = "(mannequin / built-in skeleton)"
CHARACTER_EXTS = {".glb", ".gltf", ".vrm"}


def characters_dir() -> Path:
    """角色目录：``<ComfyUI>/models/zmocap/characters``。"""
    import folder_paths

    root = Path(folder_paths.models_dir) / "zmocap" / "characters"
    root.mkdir(parents=True, exist_ok=True)
    return root


def character_choices() -> list[str]:
    """下拉框选项：内置骨架 + 角色目录里的文件。"""
    names = sorted(p.name for p in characters_dir().iterdir() if p.suffix.lower() in CHARACTER_EXTS)
    return [MANNEQUIN] + names


def resolve_character(value: str) -> str:
    """把下拉框的值解析成绝对路径；内置骨架返回空串。"""
    text = str(value or "").strip()
    if not text or text == MANNEQUIN:
        return ""
    path = Path(text).expanduser()
    if path.is_file():
        return str(path)
    candidate = characters_dir() / text
    if candidate.is_file():
        return str(candidate)
    return ""


class ZMocapCharacterPreview:
    """把角色文件（GLB/VRM）复制到 output 并返回路径，供官方 Preview3D 预览。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": (character_choices(), {"default": MANNEQUIN}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, character):
        import folder_paths

        source = resolve_character(character)
        if not source:
            raise ValueError(f"角色文件不存在：{character}（放到 {characters_dir()} 里再选）")
        outdir = Path(folder_paths.get_output_directory())
        outdir.mkdir(parents=True, exist_ok=True)
        # 统一成 .glb 后缀：VRM 本身就是 GLB 容器，官方 Preview3D 按后缀选加载器。
        target = outdir / f"zmocap_character_{Path(source).stem}.glb"
        shutil.copyfile(source, target)
        print(f"[zmocap] character preview -> {target}")
        return {"ui": {"zmocap_character": [{"path": str(target)}]}, "result": (target.name,)}
