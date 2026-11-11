"""骨骼"小棍"图层：每根骨一根细长方体，挂在对应骨骼节点下。

只做静态几何（父级是骨骼节点，动画自然带着它走），因此不增加动画数据；
前端按显示层切换：`skinned` 层看蒙皮，`bones` 层看小棍。
"""

from __future__ import annotations

import numpy as np

_UNIT_BOX = {
    # 沿 +Y 从 0 到 1，横截面 [-0.5, 0.5]（缩放后即一根棍）
    "positions": np.array([
        [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5],
        [-0.5, 1.0, -0.5], [0.5, 1.0, -0.5], [0.5, 1.0, 0.5], [-0.5, 1.0, 0.5],
    ], dtype=np.float32),
    "indices": np.array([
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2,
        2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0,
    ], dtype=np.uint16),
}

_COMPONENT_USHORT = 5123
_COMPONENT_FLOAT = 5126


def _world_of(nodes, parents, index, cache):
    if index in cache:
        return cache[index]
    local = np.eye(4)
    node = nodes[index]
    if "matrix" in node:
        local = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    else:
        translation = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
        rotation = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
        scale = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
        local[:3, :3] = _quat_mat(rotation) @ np.diag(scale)
        local[:3, 3] = translation
    parent = parents.get(index)
    world = local if parent is None else _world_of(nodes, parents, parent, cache) @ local
    cache[index] = world
    return world


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _align_y(direction: np.ndarray) -> np.ndarray:
    """把 +Y 转到 direction 的旋转矩阵。"""
    y = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(y)
    if norm < 1e-9:
        return np.eye(3)
    y = y / norm
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(target, y)
    sin = np.linalg.norm(axis)
    cos = float(target @ y)
    if sin < 1e-9:
        return np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sin
    angle = np.arctan2(sin, cos)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def add_stick_layer(appender, doc: dict, bones, parents, by_name: dict,
                    color=(0.95, 0.55, 0.15), radius_of=None) -> int:
    """给 `bones`（名字 → rest 4x4，Blender 口径）各加一根小棍，返回新增的棍数。

    `parents` 允许传 dict（下标 → 父下标）或 list（下标即子下标）；
    `head_of(name)` 给出该骨在角色空间（Z 向上）的骨骼起点，用于定位；
    棍子在骨骼节点的局部空间里摆放，所以动画会带着它一起动。
    """
    if isinstance(parents, dict):
        parent_map = dict(parents)
    else:
        parent_map = {index: int(parent) for index, parent in enumerate(parents)
                      if parent is not None and int(parent) >= 0}

    nodes = doc.setdefault("nodes", [])
    cache: dict[int, np.ndarray] = {}
    if "meshes" not in doc:
        doc["meshes"] = []
    if "materials" not in doc:
        doc["materials"] = []

    pos_view = appender.add_view(_UNIT_BOX["positions"].tobytes())
    idx_view = appender.add_view(_UNIT_BOX["indices"].tobytes())
    doc["bufferViews"][idx_view]["target"] = 34963
    doc["accessors"].append({
        "bufferView": pos_view, "componentType": _COMPONENT_FLOAT,
        "count": int(_UNIT_BOX["positions"].shape[0]), "type": "VEC3",
        "min": _UNIT_BOX["positions"].min(axis=0).tolist(),
        "max": _UNIT_BOX["positions"].max(axis=0).tolist(),
    })
    position_accessor = len(doc["accessors"]) - 1
    doc["accessors"].append({
        "bufferView": idx_view, "componentType": _COMPONENT_USHORT,
        "count": int(_UNIT_BOX["indices"].size), "type": "SCALAR",
    })
    index_accessor = len(doc["accessors"]) - 1

    doc["materials"].append({
        "name": "zmocap_stick",
        "pbrMetallicRoughness": {
            "baseColorFactor": [color[0], color[1], color[2], 1.0],
            "metallicFactor": 0.0, "roughnessFactor": 0.6,
        },
        "doubleSided": True,
    })
    material_index = len(doc["materials"]) - 1
    doc["meshes"].append({
        "name": "zmocap_stick",
        "primitives": [{
            "attributes": {"POSITION": position_accessor},
            "indices": index_accessor,
            "material": material_index,
            "mode": 4,
        }],
    })
    mesh_index = len(doc["meshes"]) - 1

    added = 0
    for name in bones:
        node_index = by_name.get(name)
        if node_index is None:
            continue
        # 骨骼轴直接取"主子的局部平移方向"——与导出坐标系无关，无需任何换算。
        parent_node = nodes[node_index]
        children = [c for c in parent_node.get("children", []) if c < len(nodes)]
        direction = None
        length = 0.0
        for child in children:
            offset = np.asarray(nodes[child].get("translation", [0, 0, 0]), dtype=np.float64)
            norm = np.linalg.norm(offset)
            if norm > length:
                direction, length = offset / norm, norm
        if direction is None or length < 1e-6:
            continue
        radius = radius_of(name) if radius_of else 0.012
        node = {
            "name": f"zmocap_stick_{name}",
            "mesh": mesh_index,
            "translation": [0.0, 0.0, 0.0],
            "rotation": _quat_xyzw(_align_y(direction)),
            "scale": [float(radius), float(length), float(radius)],
        }
        nodes.append(node)
        child_index = len(nodes) - 1
        parent_map[child_index] = node_index
        parent_node.setdefault("children", []).append(child_index)
        added += 1
    return added


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
