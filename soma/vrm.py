"""SOMA 运动 → 任意人形骨架（这里以 VRM/GLB 角色为主）的重定向。

重定向算法来自 MotionBridge（Apache-2.0，见 vendor/motionbridge/NOTICE.md）：
它以"骨骼方向 + 目标骨架自己的 rest"求解，不要求源与目标静止姿态相同，
因此不会出现"SMPL-X 与 VRM 朝向差 180 度"这类增量叠加错误。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

def _bridge():
    """延迟导入 MotionBridge（vendor 缺失时给出可执行的提示，而不是插件加载失败）。"""
    try:
        from ..vendor.motionbridge import rigmap as rigmap_module
        from ..vendor.motionbridge import retarget as retarget_module
    except Exception as exc:  # pragma: no cover - 依赖外部 vendor
        raise RuntimeError(
            "缺少 vendor/motionbridge（Apache-2.0）。请在插件目录执行 "
            "`python install.py --fetch-vendor` 或手动克隆 motionbridge 到 vendor/motionbridge。"
        ) from exc
    return rigmap_module, retarget_module
from ..nodes.retarget import (
    _Appender,
    _mat_from_quat,
    _quat_from_mat,
    _quat_mul,
    _read_glb,
    _write_glb,
)

# glTF(Y 轴向上) ↔ Blender(Z 轴向上)
_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _local_matrix(node: dict) -> np.ndarray:
    matrix = np.eye(4)
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    if "translation" in node:
        matrix[:3, 3] = node["translation"]
    if "rotation" in node:
        matrix[:3, :3] = _mat_from_quat(np.asarray(node["rotation"], dtype=np.float64))
    if "scale" in node:
        matrix[:3, :3] = matrix[:3, :3] @ np.diag(node["scale"])
    return matrix


def _parents(doc: dict) -> list:
    nodes = doc.get("nodes", [])
    parents = [None] * len(nodes)
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            if child < len(parents):
                parents[child] = index
    return parents


def build_dump(character: str | Path) -> dict:
    """从 VRM/GLB 角色读出一份"骨架转储"（Blender 口径：Z 轴向上）。"""
    doc, _ = _read_glb(Path(character))
    nodes = doc.get("nodes", [])
    if not nodes:
        raise ValueError(f"角色文件没有节点：{character}")
    parents = _parents(doc)

    world: dict[int, np.ndarray] = {}

    def world_matrix(index: int) -> np.ndarray:
        if index not in world:
            local = _local_matrix(nodes[index])
            parent = parents[index]
            world[index] = local if parent is None else world_matrix(parent) @ local
        return world[index]

    children = {i: [] for i in range(len(nodes))}
    for index, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(index)

    bones = []
    for index, node in enumerate(nodes):
        name = node.get("name") or ""
        if name.lower().endswith(("_end", ".end", "_tip", "end_site")):
            continue                                   # 末端标记节点不参与骨架识别
        matrix = world_matrix(index)
        head = _TO_Z @ matrix[:3, 3]
        tails = [_TO_Z @ world_matrix(child)[:3, 3] for child in children[index]]
        tails = [t for t in tails if np.linalg.norm(t - head) > 1e-5]
        if tails:
            tail = max(tails, key=lambda t: np.linalg.norm(t - head))
        else:
            direction = _TO_Z @ matrix[:3, 1]
            norm = np.linalg.norm(direction) or 1.0
            tail = head + direction / norm * 0.02
        # 关键：rest 必须是"骨骼基"——Y 轴沿骨骼方向（Blender/MotionBridge 的口径），
        # 不能用节点自身的旋转，否则重定向求解方向时会整体偏。
        axis_y = tail - head
        axis_y = axis_y / (np.linalg.norm(axis_y) or 1.0)
        reference_z = _TO_Z @ matrix[:3, 2]
        if abs(float(reference_z @ axis_y)) > 0.98:
            reference_z = _TO_Z @ matrix[:3, 0]
        axis_x = np.cross(axis_y, reference_z)
        axis_x = axis_x / (np.linalg.norm(axis_x) or 1.0)
        axis_z = np.cross(axis_x, axis_y)
        rest = np.eye(4)
        rest[:3, :3] = np.stack([axis_x, axis_y, axis_z], axis=1)
        rest[:3, 3] = head
        bones.append({
            "name": node.get("name") or f"node_{index}",
            "parent": None if parents[index] is None else (nodes[parents[index]].get("name") or f"node_{parents[index]}"),
            "deform": True,
            "head": head.tolist(),
            "tail": tail.tolist(),
            "rest": rest.tolist(),
            "weighted_vertices": 1,
            "girth": 0.0,
            "node": index,
        })
    return {"bones": bones, "object_matrix": np.eye(4).tolist(), "unit": "m"}


SOMA_LEFT = np.array([-1.0, 0.0, 0.0])
SOMA_FORWARD = np.array([0.0, 0.0, -1.0])
SOMA_UP = np.array([0.0, 1.0, 0.0])

# 小臂/小腿这类段长不一的关节用"方向"求，其余用相对 rest 的变化
_LIMB_ROLES = (
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _swing(a, b):
    """把方向 a 转到方向 b 的最小旋转。"""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)
        axis = _unit(np.cross(a, [1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = _unit(np.cross(a, [0.0, 1.0, 0.0]))
        return _axis_angle(axis, np.pi)
    axis = _unit(cross)
    return _axis_angle(axis, np.arccos(dot))


def _axis_angle(axis, angle):
    axis = _unit(np.asarray(axis, float))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _soma_rest(dump, role, mapping):
    """SOMA 该关节的静止方向（用角色自己的比例镜像出源的方向，见 MotionBridge 的 rationale）。"""
    bone = mapping.roles.get(role)
    if not bone:
        return None
    bone = bone[0] if isinstance(bone, (list, tuple)) else bone
    entry = next(b for b in dump["bones"] if b["name"] == bone)
    return _unit(np.asarray(entry["tail"], float) - np.asarray(entry["head"], float))


def _axis_angle(axis, angle):
    axis = _unit(np.asarray(axis, dtype=np.float64))
    x, y, z = axis
    c, s_, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s_ * z, t * x * z + s_ * y],
        [t * x * y + s_ * z, t * y * y + c, t * y * z - s_ * x],
        [t * x * z - s_ * y, t * y * z + s_ * x, t * z * z + c],
    ])


def calibrate(motion, character: str | Path) -> dict:
    """标定：源运动 + 角色 → 一次性的对齐参数（不改动源，也不改角色文件）。

    产出：坐标基底换算 ``to_rig``、朝向修正 ``yaw``、骨长比 ``scale``、
    以及 MotionBridge 的骨骼配对 ``pairs``。把标定与求解拆开之后，
    同一段动作可以配不同角色、同一个角色也可以复用同一份标定。
    """
    _, retarget_module = _bridge()
    _pairs = retarget_module._pairs

    dump = build_dump(character)
    rigmap_module, _ = _bridge()
    mapping = rigmap_module.build(dump)
    missing = mapping.missing()
    if missing:
        raise ValueError(f"角色骨架识别失败，缺少：{missing}（{mapping.notes}）")

    rig_basis = np.column_stack([mapping.left, mapping.forward, mapping.up])
    soma_basis = np.column_stack([SOMA_LEFT, SOMA_FORWARD, SOMA_UP])
    to_rig = rig_basis @ soma_basis.T

    up_axis = _unit(np.asarray(mapping.up, dtype=np.float64))
    hips_0 = np.asarray(motion.positions[0], dtype=np.float64)
    left_0 = to_rig @ (hips_0[motion.index("LeftLeg")] - hips_0[motion.index("RightLeg")])
    source_front = np.cross(_unit(left_0), up_axis)
    source_front = source_front - up_axis * float(source_front @ up_axis)
    target_front = np.asarray(mapping.forward, dtype=np.float64) - up_axis * float(mapping.forward @ up_axis)
    if np.linalg.norm(source_front) > 1e-6 and np.linalg.norm(target_front) > 1e-6:
        a, b = _unit(source_front), _unit(target_front)
        axis = np.cross(a, b)
        angle = float(np.arctan2(np.linalg.norm(axis), float(np.clip(a @ b, -1.0, 1.0))))
        yaw = _axis_angle(_unit(axis), angle) if np.linalg.norm(axis) > 1e-9 else np.eye(3)
    else:
        angle = 0.0
        yaw = np.eye(3)

    rig_head = {b["name"]: np.asarray(b["head"], dtype=np.float64) for b in dump["bones"]}
    soma_rest = np.asarray(motion.rest[:, :3, 3], dtype=np.float64)
    soma_head = {name: soma_rest[i] for i, name in enumerate(motion.names)}
    pairs = _pairs(mapping.roles, rig_head, soma_head)

    def leg_length(points):
        return float(np.mean([
            np.linalg.norm(points["LeftLeg"] - points["LeftFoot"]),
            np.linalg.norm(points["RightLeg"] - points["RightFoot"]),
        ]))

    try:
        scale = leg_length(rig_head) / leg_length(soma_head)
    except KeyError:
        scale = 1.0

    return {
        "dump": dump,
        "mapping": mapping,
        "to_rig": to_rig,
        "yaw": yaw,
        "yaw_deg": float(np.degrees(angle)),
        "scale": scale,
        "pairs": pairs,
        "bones": {b["name"]: b for b in dump["bones"]},
        "rig_head": rig_head,
        "soma_head": soma_head,
        "character": str(character),
    }


def apply_calibration(motion, calibration: dict) -> dict:
    """用标定结果求解每根骨头的世界旋转与髋部位移（纯计算，不碰文件）。"""
    to_rig = np.asarray(calibration["to_rig"], dtype=np.float64)
    yaw = np.asarray(calibration["yaw"], dtype=np.float64)
    scale = float(calibration["scale"])
    bones = calibration["bones"]
    soma_head = calibration["soma_head"]
    soma_rot = np.asarray(motion.world[:, :, :3, :3], dtype=np.float64)
    soma_anim = np.asarray(motion.positions, dtype=np.float64)

    world_rotation = {}
    for bone, joint, child, _following, _aligned in calibration["pairs"]:
        if bone not in bones or joint not in soma_head:
            continue
        rest = np.asarray(bones[bone]["rest"], dtype=np.float64)[:3, :3]
        if child in soma_head:
            j = motion.index(joint)
            c = motion.index(child)
            directions = (soma_anim[:, c] - soma_anim[:, j]) @ to_rig.T @ yaw.T
            directions = np.stack([_unit(d) for d in directions])
            ref = np.tile(np.asarray(rest[:, 2], dtype=np.float64), (len(directions), 1))
            zaxis = ref - directions * np.sum(ref * directions, axis=1, keepdims=True)
            bad = np.linalg.norm(zaxis, axis=1) < 1e-6
            if np.any(bad):
                zaxis[bad] = np.cross(directions[bad], np.array([1.0, 0.0, 0.0]))
            zaxis = np.stack([_unit(z) for z in zaxis])
            xaxis = np.cross(directions, zaxis)
            world_rotation[bone] = np.stack([xaxis, directions, zaxis], axis=2)
        else:
            delta = yaw @ to_rig @ soma_rot[:, motion.index(joint)] @ to_rig.T @ yaw.T
            world_rotation[bone] = delta @ rest

    hips_index = motion.index("Hips")
    hips = (soma_anim[:, hips_index] - soma_anim[0, hips_index]) @ to_rig.T @ yaw.T * scale
    return {
        "world_rotation": world_rotation,
        "hips_offset": hips,
        "frames": motion.frames,
        "fps": motion.fps,
        "scale": scale,
        "pairs": len(calibration["pairs"]),
        "yaw_deg": float(calibration.get("yaw_deg", 0.0)),
    }


def retarget_to_rig(motion, character: str | Path) -> dict:
    """一次性完成标定 + 求解（保留旧接口，等价于 calibrate → apply_calibration）。"""
    calibration = calibrate(motion, character)
    result = apply_calibration(motion, calibration)
    result["dump"] = calibration["dump"]
    result["rigmap"] = calibration["mapping"]
    result["to_rig"] = calibration["to_rig"]
    return result


def write_animation(character: str | Path, out_path: str | Path, retargeted: dict, fps: float,
                    sticks: bool = True) -> Path:
    """把重定向结果写成带动画的 GLB（角色网格原样保留）。

    world_rotation 是"骨骼基"下的世界旋转；节点空间的对应旋转为
        node_world = B(t) @ B_rest⁻¹ @ W_rest
    再按层级求 local = parent_world(t)⁻¹ @ world(t)，因此对带缩放节点同样成立。
    """
    character, out_path = Path(character), Path(out_path)
    doc, binary = _read_glb(character)
    nodes = doc.get("nodes", [])
    parents = _parents(doc)
    by_name = {node.get("name"): index for index, node in enumerate(nodes) if node.get("name")}

    world_rotation = retargeted["world_rotation"]                 # bone -> (T,3,3)
    hips_offset = np.asarray(retargeted.get("hips_offset", np.zeros((0, 3))), dtype=np.float64)
    bones = {b["name"]: b for b in retargeted["dump"]["bones"]}
    frames = int(retargeted["frames"])
    fps = float(fps or retargeted.get("fps") or 30.0)
    times = np.arange(frames, dtype=np.float64) / fps

    rest_world = {}

    def rest_world_of(index: int) -> np.ndarray:
        if index not in rest_world:
            local = _local_matrix(nodes[index])
            parent = parents[index]
            rest_world[index] = local if parent is None else rest_world_of(parent) @ local
        return rest_world[index]

    for index in range(len(nodes)):
        rest_world_of(index)

    node_order, seen = [], set()
    while len(node_order) < len(nodes):
        progressed = False
        for index in range(len(nodes)):
            if index in seen:
                continue
            if parents[index] is None or parents[index] in seen:
                node_order.append(index)
                seen.add(index)
                progressed = True
        if not progressed:
            break

    animated, hips_node = {}, None
    for bone_name, rotation in world_rotation.items():
        node_index = by_name.get(bone_name)
        if node_index is None:
            continue
        rest = np.asarray(bones[bone_name]["rest"], dtype=np.float64)
        rest_world_rotation = _pure_rotation(rest_world[node_index][:3, :3])
        animated[node_index] = rotation @ np.linalg.inv(rest[:3, :3]) @ rest_world_rotation
        if bones[bone_name].get("parent") is None or bone_name == retargeted["rigmap"].roles.get("Hips"):
            hips_node = node_index

    hips_name = retargeted["rigmap"].roles.get("Hips")
    hips_name = hips_name[0] if isinstance(hips_name, (list, tuple)) else hips_name
    if hips_name:
        hips_node = by_name.get(hips_name)

    world_rot, world_trans = {}, {}
    for node_index in node_order:
        rest = rest_world[node_index]
        world_rot[node_index] = animated.get(node_index, np.broadcast_to(_pure_rotation(rest[:3, :3]), (frames, 3, 3)))
        translation = np.broadcast_to(rest[:3, 3], (frames, 3)).copy()
        if node_index == hips_node and hips_offset.size:
            translation = translation + hips_offset @ _TO_Z
        world_trans[node_index] = translation

    local_rot, local_trans = {}, {}
    for node_index in node_order:
        parent = parents[node_index]
        if parent is None:
            local_rot[node_index] = world_rot[node_index]
            local_trans[node_index] = world_trans[node_index]
        else:
            local_rot[node_index] = np.einsum("tji,tjk->tik", world_rot[parent], world_rot[node_index])
            local_trans[node_index] = np.einsum("tji,tj->ti", world_rot[parent], world_trans[node_index] - world_trans[parent])

    appender = _Appender(doc, binary)
    time_accessor = appender.add_floats(times, "SCALAR", minmax=True)
    samplers, channels = [], []
    for node_index in node_order:
        if node_index not in animated:
            continue
        # 注意：时间轴由 sampler.input 提供，output 只能是纯旋转（VEC4×帧数）
        values = _quat_from_mat(local_rot[node_index]).astype(np.float32)
        sampler = len(samplers)
        samplers.append({"input": time_accessor, "interpolation": "LINEAR", "output": appender.add_floats(values, "VEC4")})
        channels.append({"sampler": sampler, "target": {"node": node_index, "path": "rotation"}})

    if hips_node is not None and hips_node in local_trans:
        values = np.asarray(local_trans[hips_node], dtype=np.float32)
        sampler = len(samplers)
        samplers.append({"input": time_accessor, "interpolation": "LINEAR", "output": appender.add_floats(values, "VEC3")})
        channels.append({"sampler": sampler, "target": {"node": hips_node, "path": "translation"}})

    if sticks:
        from ..nodes.sticks import add_stick_layer

        try:
            stick_bones = {name: bones[name] for name in retargeted["world_rotation"] if name in bones}
            count = add_stick_layer(appender, doc, stick_bones, parents, by_name)
            print(f"[zmocap] 加骨骼小棍 {count} 根")
        except Exception as exc:                       # 小棍只是显示层，失败不该毁掉整份 GLB
            print(f"[zmocap] 小棍图层失败（{exc}），跳过")

    doc.setdefault("animations", [])
    doc["animations"].append({"name": "zmocap", "samplers": samplers, "channels": channels})
    appender.finish()
    return _write_glb(out_path, doc, appender.binary)


def _pure_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64)[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _world_matrix(nodes, parents, index):
    local = _local_matrix(nodes[index])
    parent = parents[index]
    return local if parent is None else _world_matrix(nodes, parents, parent) @ local


def _quat_to_xyzw(quats: np.ndarray) -> np.ndarray:
    """MotionBridge 的 (w, x, y, z) → glTF 的 (x, y, z, w)。"""
    quats = np.asarray(quats, dtype=np.float64)
    return np.stack([quats[..., 1], quats[..., 2], quats[..., 3], quats[..., 0]], axis=-1)


def _quat_to_gltf(quats: np.ndarray) -> np.ndarray:
    return np.asarray(quats, dtype=np.float64)


_Z_TO_Y = _TO_Z.T
