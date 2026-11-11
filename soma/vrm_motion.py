"""SOMA 运动 → 标准 VRM humanoid 运动（纯格式转换，与具体角色无关）。

输出的是"在一套规范骨架上的运动"：
    {"skeleton": "vrm", "fps", "frames", "names", "parents",
     "world_rotation": (T,B,3,3), "transl": (T,3)}

谁把它穿上身（角色绑定）由绑定节点负责——它用角色自己的 rest 吸收 A-pose、
身高、骨长差异，并把坐标写进 GLB。
"""

from __future__ import annotations

import numpy as np

from . import humanoid

# SOMA 关节 ← → humanoid 骨骼（标准名）
_BONES = {
    "hips": "Hips",
    "spine": "Spine1",
    "chest": "Spine2",
    "upperChest": "Chest",
    "neck": "Neck1",
    "head": "Head",
}
for side in ("left", "right"):
    cap = "Left" if side == "left" else "Right"
    _BONES[f"{side}Shoulder"] = f"{cap}Shoulder"
    _BONES[f"{side}UpperArm"] = f"{cap}Arm"
    _BONES[f"{side}LowerArm"] = f"{cap}ForeArm"
    _BONES[f"{side}Hand"] = f"{cap}Hand"
    _BONES[f"{side}UpperLeg"] = f"{cap}Leg"
    _BONES[f"{side}LowerLeg"] = f"{cap}Shin"
    _BONES[f"{side}Foot"] = f"{cap}Foot"
    _BONES[f"{side}Toes"] = f"{cap}ToeBase"
    for finger in ("Index", "Middle", "Ring", "Little"):
        for level, suffix in (("Proximal", "1"), ("Intermediate", "2"), ("Distal", "3")):
            _BONES[f"{side}{finger}{level}"] = f"{cap}Hand{finger}{suffix}"
    for level, suffix in (("Proximal", "1"), ("Intermediate", "2"), ("Distal", "3")):
        _BONES[f"{side}Thumb{level}"] = f"{cap}HandThumb{suffix}"

# 无子骨可用的叶子骨：用该 SOMA 关节自身的旋转推方向
_LEAF_FALLBACK = {
    "leftToes": "LeftToeEnd",
    "rightToes": "RightToeEnd",
    "leftIndexDistal": "LeftHandIndexEnd",
    "leftMiddleDistal": "LeftHandMiddleEnd",
    "leftRingDistal": "LeftHandRingEnd",
    "leftLittleDistal": "LeftHandLittleEnd",
    "leftThumbDistal": "LeftHandThumbEnd",
    "rightIndexDistal": "RightHandIndexEnd",
    "rightMiddleDistal": "RightHandMiddleEnd",
    "rightRingDistal": "RightHandRingEnd",
    "rightLittleDistal": "RightHandLittleEnd",
    "rightThumbDistal": "RightHandThumbEnd",
}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def soma_to_vrm(motion) -> dict:
    """SOMA 运动 → VRM humanoid 运动（不涉及任何角色文件）。"""
    names = list(humanoid.NAMES)
    parents = list(humanoid.PARENTS)
    rest = humanoid.rest_matrices()
    rest_dir = humanoid.bone_directions()
    frames = motion.frames
    positions = np.asarray(motion.positions, dtype=np.float64)

    children = {name: [] for name in names}
    for index, parent in enumerate(parents):
        if parent:
            children[parent].append(index)

    available = set(motion.names)

    def joint_index(name):
        return motion.index(name) if name in available else None

    world_rotation = np.zeros((frames, len(names), 3, 3))
    for index, bone in enumerate(names):
        joint = _BONES.get(bone)
        if joint not in available:
            # 源里没有这根骨（例如 SOMA 未含手指）→ 保持标准 rest，不猜
            world_rotation[:, index] = rest[index, :3, :3]
            continue
        j = motion.index(joint)
        source_child = None
        for child_index in children[bone]:
            child_joint = _BONES.get(names[child_index])
            if child_joint in available:
                source_child = child_joint
                break
        if source_child is not None:
            c = motion.index(source_child)
            directions = np.stack([_unit(d) for d in (positions[:, c] - positions[:, j])])
        elif bone in _LEAF_FALLBACK and _LEAF_FALLBACK[bone] in available:
            c = motion.index(_LEAF_FALLBACK[bone])
            directions = np.stack([_unit(d) for d in (positions[:, c] - positions[:, j])])
        else:
            directions = np.einsum("tij,j->ti", np.asarray(motion.world[:frames, j, :3, :3]), rest_dir[index])
            directions = np.stack([_unit(d) for d in directions])
        # 以标准 rest 的 Z 轴作 roll 参考，保证骨链扭转稳定
        reference = np.tile(rest[index, :3, 2], (frames, 1))
        zaxis = reference - directions * np.sum(reference * directions, axis=1, keepdims=True)
        weak = np.linalg.norm(zaxis, axis=1) < 1e-6
        if np.any(weak):
            zaxis[weak] = np.cross(directions[weak], np.array([1.0, 0.0, 0.0]))
        zaxis = np.stack([_unit(z) for z in zaxis])
        xaxis = np.cross(directions, zaxis)
        world_rotation[:, index] = np.stack([xaxis, directions, zaxis], axis=2)

    hips = motion.index("Hips")
    transl = positions[:, hips] - positions[0, hips]
    return {
        "skeleton": "vrm",
        "fps": motion.fps,
        "frames": frames,
        "names": names,
        "parents": parents,
        "rest": rest,
        "world_rotation": world_rotation,
        "transl": transl,
    }
