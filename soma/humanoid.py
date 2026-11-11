"""标准 VRM humanoid 骨架（T-pose）。

VRM 的人形骨骼是**规范化的**：名字、父子关系、朝向口径都固定，所以"把动作表示成
VRM humanoid 骨骼上的运动"这一步与具体角色无关。具体角色只在最后绑定/导出时才需要，
由绑定节点用角色自己的 rest 去吸收 A-pose、身高、骨长差异。

坐标：Y 向上，正面 -Z（glTF/VRM 口径），单位米，参考身高 1.5 m。
"""

from __future__ import annotations

import numpy as np

# (bone, parent, head position) —— T-pose，左臂沿 +X，腿沿 -Y，脚趾朝 -Z
_TABLE = [
    ("hips", None, (0.0, 0.90, 0.0)),
    ("spine", "hips", (0.0, 0.98, 0.0)),
    ("chest", "spine", (0.0, 1.09, 0.0)),
    ("upperChest", "chest", (0.0, 1.20, 0.0)),
    ("neck", "upperChest", (0.0, 1.32, 0.0)),
    ("head", "neck", (0.0, 1.42, 0.0)),
]

for side, sx in (("left", 1.0), ("right", -1.0)):
    _TABLE += [
        (f"{side}Shoulder", "upperChest", (0.04 * sx, 1.28, 0.0)),
        (f"{side}UpperArm", f"{side}Shoulder", (0.16 * sx, 1.28, 0.0)),
        (f"{side}LowerArm", f"{side}UpperArm", (0.42 * sx, 1.28, 0.0)),
        (f"{side}Hand", f"{side}LowerArm", (0.68 * sx, 1.28, 0.0)),
        (f"{side}UpperLeg", "hips", (0.09 * sx, 0.88, 0.0)),
        (f"{side}LowerLeg", f"{side}UpperLeg", (0.09 * sx, 0.48, 0.0)),
        (f"{side}Foot", f"{side}LowerLeg", (0.09 * sx, 0.07, 0.0)),
        (f"{side}Toes", f"{side}Foot", (0.09 * sx, 0.02, -0.08)),
        (f"{side}ThumbProximal", f"{side}Hand", (0.69 * sx, 1.27, 0.01)),
        (f"{side}ThumbIntermediate", f"{side}ThumbProximal", (0.70 * sx, 1.27, 0.02)),
        (f"{side}ThumbDistal", f"{side}ThumbIntermediate", (0.71 * sx, 1.27, 0.03)),
    ]
    for finger, off in (("Index", 0.0), ("Middle", 0.0), ("Ring", 0.0), ("Little", 0.0)):
        pass
    for index, finger in enumerate(("Index", "Middle", "Ring", "Little")):
        z = 0.02 - 0.014 * index
        _TABLE += [
            (f"{side}{finger}Proximal", f"{side}Hand", (0.70 * sx, 1.28, z)),
            (f"{side}{finger}Intermediate", f"{side}{finger}Proximal", (0.72 * sx, 1.28, z)),
            (f"{side}{finger}Distal", f"{side}{finger}Intermediate", (0.74 * sx, 1.28, z)),
        ]

NAMES = [row[0] for row in _TABLE]
PARENTS = [row[1] for row in _TABLE]
POSITIONS = np.array([row[2] for row in _TABLE], dtype=np.float64)
INDEX = {name: i for i, name in enumerate(NAMES)}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def bone_directions() -> np.ndarray:
    """每根骨的朝向（指向子骨；无子骨的叶子骨指向 +Y）。"""
    out = np.tile(np.array([0.0, 1.0, 0.0]), (len(NAMES), 1))
    for index, name in enumerate(NAMES):
        kids = [j for j, parent in enumerate(PARENTS) if parent == name]
        if not kids:
            continue
        best = max(kids, key=lambda j: np.linalg.norm(POSITIONS[j] - POSITIONS[index]))
        vector = POSITIONS[best] - POSITIONS[index]
        if np.linalg.norm(vector) > 1e-6:
            out[index] = _unit(vector)
    return out


def bone_matrix(head, direction) -> np.ndarray:
    """以"局部 Y = 骨向"构造骨骼矩阵的旋转部分（与 Blender 一致）。"""
    y = _unit(np.asarray(direction, dtype=np.float64))
    reference = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(reference, y))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def rest_matrices() -> np.ndarray:
    """标准 humanoid 的静止骨骼矩阵 (B, 4, 4)。"""
    directions = bone_directions()
    out = np.zeros((len(NAMES), 4, 4))
    for i in range(len(NAMES)):
        out[i, :3, :3] = bone_matrix(POSITIONS[i], directions[i])
        out[i, :3, 3] = POSITIONS[i]
        out[i, 3, 3] = 1.0
    return out
