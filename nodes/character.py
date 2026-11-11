"""ZMocap VRM Loader：读取角色文件（VRM / GLB / GLTF），输出可复用的角色对象。

职责单一：**只做"载入角色"**，不做动作、不做渲染。
- 校验文件、解析骨骼映射（Mixamo / VRM humanoid 命名）、统计网格
- 复制一份到 output 目录并给出可被官方 ``Preview3D`` 播放的相对路径
  （这样上传的角色本身也能直接预览）
- 输出 ``ZMOCAP_CHARACTER``，供 ``zmocap Rig (GLB)`` 使用

上传按钮在 ``web/js/zmocap.js``：走官方 ``POST /upload/image`` 存到 ``input/``。
"""

from __future__ import annotations

import shutil
from pathlib import Path

MANNEQUIN = "(none / built-in skeleton)"
CHARACTER_EXTS = {".glb", ".gltf", ".vrm"}


def characters_dir() -> Path:
    """角色目录：``<ComfyUI>/models/zmocap/characters``。"""
    import folder_paths

    root = Path(folder_paths.models_dir) / "zmocap" / "characters"
    root.mkdir(parents=True, exist_ok=True)
    return root


def input_dir() -> Path:
    """ComfyUI 的 input 目录，前端上传的角色文件会落在这里。"""
    import folder_paths

    root = Path(folder_paths.get_input_directory())
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scan(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [p.name for p in root.iterdir() if p.suffix.lower() in CHARACTER_EXTS]


def character_choices() -> list[str]:
    """下拉框选项：无角色 + 角色目录 + 前端上传到 input/ 的文件。"""
    names: list[str] = []
    for root in (characters_dir(), input_dir()):
        for name in sorted(_scan(root)):
            if name not in names:
                names.append(name)
    return [MANNEQUIN] + names


def resolve_character(value: str) -> str:
    """把下拉框/路径解析成绝对路径；无角色返回空串。"""
    text = str(value or "").strip()
    if not text or text == MANNEQUIN:
        return ""
    path = Path(text).expanduser()
    if path.is_file():
        return str(path)
    for root in (characters_dir(), input_dir()):
        candidate = root / text
        if candidate.is_file():
            return str(candidate)
    return ""


class ZMocapVRMLoader:
    """载入角色文件，输出 ``ZMOCAP_CHARACTER``（同时给出可预览的 GLB 相对路径）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": (character_choices(), {"default": MANNEQUIN}),
            }
        }

    RETURN_TYPES = ("ZMOCAP_CHARACTER", "STRING")
    RETURN_NAMES = ("character", "glb_path")
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, character):
        import folder_paths

        from .retarget import _bone_map, _read_glb

        source = resolve_character(character)
        if not source:
            raise ValueError(f"角色文件不存在：{character}（放到 {characters_dir()} 或上传）")

        doc, _ = _read_glb(Path(source))
        mapping = _bone_map(doc)
        if not mapping:
            print(f"[zmocap] {Path(source).name}: 没匹配到 Mixamo/VRM 骨骼名，Rig 会回退到自带骨架")

        outdir = Path(folder_paths.get_output_directory())
        outdir.mkdir(parents=True, exist_ok=True)
        # 统一成 .glb：VRM 本身就是 GLB 容器，官方 Preview3D 按后缀挑加载器。
        target = outdir / f"zmocap_character_{Path(source).stem}.glb"
        if Path(source).resolve() != target.resolve():
            shutil.copyfile(source, target)

        meshes = len(doc.get("meshes", []))
        print(f"[zmocap] character {Path(source).name}: {len(doc.get('nodes', []))} nodes, "
              f"{meshes} meshes, {len(mapping)}/21 humanoid bones -> {target.name}")

        payload = {
            "path": source,
            "name": Path(source).name,
            "preview": target.name,
            "nodes": len(doc.get("nodes", [])),
            "meshes": meshes,
            "bones": len(mapping),
        }
        return {"ui": {"zmocap_character": [payload]}, "result": (payload, target.name)}
