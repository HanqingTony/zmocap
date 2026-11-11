"""把标准 humanoid 运动穿到具体角色上——这是**唯一**需要角色文件的环节。

运动本身（`skeleton == "vrm"`）描述的是"规范骨架上发生了什么"；绑定要用角色自己的
rest 基去吸收 A-pose、身高与骨长差异：

    B_can  标准 humanoid 的静止骨骼基
    B_char 角色自己的静止骨骼基
    W_char(t) = A(t) @ B_can⁻¹ @ B_char          （t=静止时 A=B_can ⇒ W_char=B_char）
"""

from __future__ import annotations

import types

import numpy as np

from . import humanoid
from .vrm import build_dump, write_animation

_PREFIXES = ("j_bip_", "mixamorig:", "mixamorig_", "armature|", "armature:")

_ALIAS: dict[str, str] = {
    "chips": "hips", "cspine": "spine", "cchest": "chest", "cupperchest": "upperChest",
    "cneck": "neck", "chead": "head",
}
for _side, _cap in (("l", "left"), ("r", "right")):
    _ALIAS.update({
        f"{_side}shoulder": f"{_cap}Shoulder",
        f"{_side}upperarm": f"{_cap}UpperArm",
        f"{_side}arm": f"{_cap}UpperArm",
        f"{_side}lowerarm": f"{_cap}LowerArm",
        f"{_side}forearm": f"{_cap}LowerArm",
        f"{_side}hand": f"{_cap}Hand",
        f"{_side}upperleg": f"{_cap}UpperLeg",
        f"{_side}upleg": f"{_cap}UpperLeg",
        f"{_side}thigh": f"{_cap}UpperLeg",
        f"{_side}lowerleg": f"{_cap}LowerLeg",
        f"{_side}leg": f"{_cap}LowerLeg",
        f"{_side}shin": f"{_cap}LowerLeg",
        f"{_side}foot": f"{_cap}Foot",
        f"{_side}toebase": f"{_cap}Toes",
        f"{_side}toes": f"{_cap}Toes",
    })
    for _finger, _tag in (("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("little", "Little")):
        for _level, _suffix in ((1, "Proximal"), (2, "Intermediate"), (3, "Distal")):
            _ALIAS[f"{_side}{_finger}{_level}"] = f"{_cap}{_tag}{_suffix}"
            _ALIAS[f"{_side}hand{_finger}{_level}"] = f"{_cap}{_tag}{_suffix}"
    for _level, _suffix in ((1, "Proximal"), (2, "Intermediate"), (3, "Distal")):
        _ALIAS[f"{_side}thumb{_level}"] = f"{_cap}Thumb{_suffix}"
        _ALIAS[f"{_side}handthumb{_level}"] = f"{_cap}Thumb{_suffix}"


def _normalize(name: str) -> str:
    text = str(name or "").strip()
    low = text.lower()
    for prefix in _PREFIXES:
        if low.startswith(prefix):
            text = text[len(prefix):]
            low = text.lower()
            break
    for separator in (" ", "_", "-", "."):
        text = text.replace(separator, "")
    return text.lower()


def humanoid_of(bone_name: str) -> str | None:
    """角色骨骼名 → 标准 humanoid 名（支持 VRM/VRoid 的 J_Bip_* 与 Mixamo 命名）。"""
    return _ALIAS.get(_normalize(bone_name))


def _leg_length(heads: dict) -> float:
    values = []
    for side in ("Left", "Right"):
        try:
            values.append(np.linalg.norm(heads[f"{side}Leg"] - heads[f"{side}Foot"]))
        except KeyError:
            continue
    return float(np.mean(values)) if values else 1.0


def bind(vrm_motion: dict, character: str) -> dict:
    """把 humanoid 运动绑到角色骨架上，返回可直接写 GLB 的结构。"""
    dump = build_dump(character)
    rest_canonical = humanoid.rest_matrices()
    index_canonical = {name: i for i, name in enumerate(humanoid.NAMES)}
    animation = np.asarray(vrm_motion["world_rotation"], dtype=np.float64)

    world_rotation = {}
    matched_humanoid = set()
    bone_head: dict[str, np.ndarray] = {}
    for bone in dump["bones"]:
        bone_head[bone["name"]] = np.asarray(bone["head"], dtype=np.float64)
        role = humanoid_of(bone["name"])
        if role is None:
            continue
        i = index_canonical[role]
        rest_char = np.asarray(bone["rest"], dtype=np.float64)[:3, :3]
        transfer = rest_canonical[i][:3, :3].T @ rest_char
        world_rotation[bone["name"]] = animation[:, i] @ transfer
        matched_humanoid.add(role)

    if not world_rotation:
        raise ValueError(f"角色 {character} 里没有识别到任何 humanoid 骨骼")

    matched_heads = {name: head for name, head in bone_head.items() if humanoid_of(name)}
    canonical_heads = {humanoid.NAMES[i]: humanoid.POSITIONS[i] for i in range(len(humanoid.NAMES))}
    scale = _leg_length(matched_heads) / _leg_length(canonical_heads)
    transl = np.asarray(vrm_motion["transl"], dtype=np.float64) * scale

    hips_bone = next((name for name in matched_heads if humanoid_of(name) == "hips"), None)
    rigmap = types.SimpleNamespace(roles={"Hips": hips_bone})
    missing = [name for name in humanoid.NAMES if name not in matched_humanoid]
    return {
        "dump": dump,
        "world_rotation": world_rotation,
        "hips_offset": transl,
        "frames": int(vrm_motion["frames"]),
        "fps": float(vrm_motion["fps"]),
        "scale": scale,
        "rigmap": rigmap,
        "report": {
            "bones": len(world_rotation),
            "scale": round(float(scale), 4),
            "unmapped": missing[:12],
            "unmapped_count": len(missing),
        },
    }


def write_character_glb(vrm_motion: dict, character: str, out_path, fps: float | None = None,
                        sticks: bool = True):
    """绑定并写出带动画的 GLB；返回 (路径, 绑定报告)。"""
    retargeted = bind(vrm_motion, character)
    if "dump" not in retargeted or "world_rotation" not in retargeted:
        raise RuntimeError("绑定结果缺少写 GLB 需要的数据")
    path = write_animation(character, out_path, retargeted, fps or vrm_motion["fps"], sticks=sticks)
    return path, retargeted["report"]


def write_humanoid_glb(vrm_motion: dict, out_path) -> tuple[object, dict]:
    """没接角色时，用标准 humanoid 骨架（含完整五指）写一份可预览的 GLB。"""
    import json
    import struct

    rest = humanoid.rest_matrices()
    names = list(humanoid.NAMES)
    parents = list(humanoid.PARENTS)
    index_of = {name: i for i, name in enumerate(names)}
    animation = np.asarray(vrm_motion["world_rotation"], dtype=np.float64)
    fps = float(vrm_motion["fps"])
    frames = int(vrm_motion["frames"])

    # 节点：局部静止变换 + 层级
    nodes = []
    for i, name in enumerate(names):
        parent = parents[i]
        local = np.linalg.inv(rest[index_of[parent]]) @ rest[i] if parent else rest[i]
        nodes.append({
            "name": name,
            "translation": [float(v) for v in local[:3, 3]],
            "rotation": _quat_xyzw(local[:3, :3]),
            "children": [j for j in range(len(names)) if parents[j] == name],
        })

    # 动画：把世界旋转换成节点局部旋转
    locals_world = np.zeros((frames, len(names), 3, 3))
    for i, name in enumerate(names):
        rest_rotation = rest[i][:3, :3]
        old = np.asarray(world_rotation_of(animation, name, index_of, rest_rotation))
        parent = parents[i]
        if parent:
            pindex = index_of[parent]
            parent_world = np.asarray(world_rotation_of(animation, parent, index_of, rest[index_of[parent]][:3, :3]))
            locals_world[:, i] = np.einsum("tji,tjk->tik", parent_world, old)
        else:
            locals_world[:, i] = old

    times = np.arange(frames, dtype=np.float64) / fps
    samplers, channels = [], []
    binary = bytearray()

    def add_floats(values: np.ndarray, type_: str) -> int:
        view = {"SCALAR": 1, "VEC4": 4}[type_]
        data = np.asarray(values, dtype=np.float32).reshape(-1, view)
        offset = len(binary)
        binary.extend(data.tobytes())
        while len(binary) % 4:
            binary.append(0)
        accessors.append({
            "bufferView": len(views), "componentType": 5126, "count": int(data.shape[0]),
            "type": type_, "min": data.min(0).tolist(), "max": data.max(0).tolist(),
        })
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": int(data.nbytes)})
        return len(accessors) - 1

    accessors: list[dict] = []
    views: list[dict] = []
    time_accessor = add_floats(times, "SCALAR")
    for i in range(len(names)):
        quats = np.stack([_quat_xyzw(m) for m in locals_world[:, i]])
        sampler = {"input": time_accessor, "output": add_floats(quats, "VEC4"), "interpolation": "LINEAR"}
        samplers.append(sampler)
        channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": "rotation"}})

    doc = {
        "asset": {"version": "2.0", "generator": "zmocap"},
        "scene": 0,
        "scenes": [{"nodes": [index_of["hips"]]}],
        "nodes": nodes,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "animations": [{"name": "zmocap", "samplers": samplers, "channels": channels}],
    }
    _write_glb_bytes(out_path, doc, binary)
    return out_path, {"bones": len(names), "frames": frames}


def world_rotation_of(animation, name, index_of, rest_rotation):
    """规范运动里该骨的世界旋转；未参与动画的骨用静止旋转。"""
    i = index_of[name]
    return animation[:, i]


def _quat_xyzw(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _write_glb_bytes(path, doc: dict, binary: bytes) -> None:
    import json
    import struct
    from pathlib import Path

    doc["buffers"][0]["byteLength"] = len(binary)
    payload = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    blob = bytes(binary)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(payload) + 8 + len(blob)
    out = bytearray()
    out += struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(payload), b"JSON") + payload
    out += struct.pack("<I4s", len(blob), b"BIN\x00") + blob
    Path(path).write_bytes(bytes(out))
