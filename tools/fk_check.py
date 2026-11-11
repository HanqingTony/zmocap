"""离线 FK 校验：把绑定生成的 GLB 动画还原成世界姿态，与 motion 的关节朝向比对。

不依赖渲染、不依赖 GPU：读 GLB → 按 sampler 采样到每帧 → 场景图前向运动学 →
取「骨骼方向向量」与 SMPL-X 关节方向向量做夹角（尺度无关），给出分层误差。

用法::

    python -m zmocap.tools.fk_check --motion motion.npz --glb output/zmocap_rig.glb [--frames 0,30,60]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- glTF 读取


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"不是 GLB：{path}")
    offset, doc, binary = 12, None, b""
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == 0x4E4F534A:
            doc = json.loads(payload.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = payload
        offset += 8 + chunk_length + ((4 - chunk_length % 4) % 4 if chunk_length % 4 else 0)
    return doc, binary


_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPES = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8, 5122: np.int16}


def accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = _DTYPES[acc["componentType"]]
    width = _COMPONENTS[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    stride = view.get("byteStride") or (width * np.dtype(dtype).itemsize)
    if stride == width * np.dtype(dtype).itemsize:
        array = np.frombuffer(binary, dtype=dtype, count=count * width, offset=start)
        return array.reshape(count, width).astype(np.float64)
    out = np.empty((count, width), dtype=np.float64)
    for row in range(count):
        base = start + row * stride
        out[row] = np.frombuffer(binary, dtype=dtype, count=width, offset=base)
    return out


# ---------------------------------------------------------------- 数学工具


def quat_to_mat(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    norm = np.where(norm == 0, 1.0, norm)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
    positive = trace > 0
    s = np.sqrt(np.where(positive, trace + 1.0, 1.0)) * 2.0
    q[..., 3] = np.where(positive, 0.25 * s, 0.0)
    q[..., 0] = np.where(positive, (m[..., 2, 1] - m[..., 1, 2]) / s, 0.0)
    q[..., 1] = np.where(positive, (m[..., 0, 2] - m[..., 2, 0]) / s, 0.0)
    q[..., 2] = np.where(positive, (m[..., 1, 0] - m[..., 0, 1]) / s, 0.0)
    for cond, (i, j, k, w) in (
        (~positive & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2]), (0, 1, 2, 3)),
        (~positive & ~(m[..., 0, 0] > m[..., 1, 1]) & (m[..., 1, 1] > m[..., 2, 2]), (1, 2, 0, 3)),
        (~positive & ~(m[..., 0, 0] > m[..., 1, 1]) & ~(m[..., 1, 1] > m[..., 2, 2]), (2, 0, 1, 3)),
    ):
        s2 = np.sqrt(np.where(cond, 1.0 + m[..., i, i] - m[..., j, j] - m[..., k, k], 1.0)) * 2.0
        q[..., i] = np.where(cond, 0.25 * s2, q[..., i])
        q[..., j] = np.where(cond, (m[..., j, i] + m[..., i, j]) / s2, q[..., j])
        q[..., k] = np.where(cond, (m[..., k, i] + m[..., i, k]) / s2, q[..., k])
        q[..., 3] = np.where(cond, (m[..., k, j] - m[..., j, k]) / s2, q[..., 3])
    return q


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, angle, out=np.zeros_like(aa), where=angle > 1e-9)
    half = angle * 0.5
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


# ---------------------------------------------------------------- 采样 + FK


def sample_animation(doc: dict, binary: bytes, frame: int, fps: float) -> dict:
    """返回 {node_index: (translation, rotation_quat)}，未被动画覆盖的节点用 rest 值。"""
    nodes = doc.get("nodes", [])
    state = {}
    for index, node in enumerate(nodes):
        state[index] = (
            np.asarray(node.get("translation") or [0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray(node.get("rotation") or [0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray(node.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64),
        )
    time = frame / float(fps)
    for animation in doc.get("animations", []):
        for channel in animation["channels"]:
            node_index = channel["target"]["node"]
            path = channel["target"]["path"]
            sampler = animation["samplers"][channel["sampler"]]
            times = accessor(doc, binary, sampler["input"]).reshape(-1)
            values = accessor(doc, binary, sampler["output"])
            if len(times) == 0:
                continue
            clamped = float(np.clip(time, times[0], times[-1]))
            if len(times) == 1:
                value = values[0]
            else:
                i = int(np.searchsorted(times, clamped, side="right") - 1)
                i = min(max(i, 0), len(times) - 2)
                span = times[i + 1] - times[i]
                weight = 0.0 if span <= 0 else (clamped - times[i]) / span
                if path == "rotation":
                    a, b = values[i], values[i + 1]
                    if float(np.dot(a, b)) < 0:
                        b = -b
                    value = a * (1 - weight) + b * weight
                    value = value / max(np.linalg.norm(value), 1e-9)
                else:
                    value = values[i] * (1 - weight) + values[i + 1] * weight
            translation, rotation, scale = state[node_index]
            if path == "translation":
                translation = np.asarray(value, dtype=np.float64)
            elif path == "rotation":
                rotation = np.asarray(value, dtype=np.float64)
            elif path == "scale":
                scale = np.asarray(value, dtype=np.float64)
            state[node_index] = (translation, rotation, scale)
    return state


def world_positions(doc: dict, state: dict) -> dict:
    nodes = doc.get("nodes", [])
    parent = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []) or []:
            parent[int(child)] = index

    world = {}

    def resolve(index: int) -> np.ndarray:
        if index in world:
            return world[index]
        translation, rotation, scale = state[index]
        local = np.eye(4)
        local[:3, :3] = quat_to_mat(rotation) @ np.diag(scale)
        local[:3, 3] = translation
        if index in parent:
            value = resolve(parent[index]) @ local
        else:
            value = local
        world[index] = value
        return value

    for index in range(len(nodes)):
        resolve(index)
    return {index: matrix[:3, 3] for index, matrix in world.items()}


# ---------------------------------------------------------------- 主流程


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, help="ZMocapExport 写出的 .npz")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--frames", default="0,15,30,45,60")
    parser.add_argument("--json", default="")
    parser.add_argument("--self", action="store_true",
                        help="自洽性检查：角色世界朝向 vs rest×SMPL-X 世界增量")
    args = parser.parse_args()

    from zmocap.nodes.retarget import (SMPLX_NAMES, _bone_map, _rest_world,
                                       _smplx_rest_joints, _world_deltas, local_rotations)

    motion = dict(np.load(args.motion))
    joints = np.asarray(motion["joints"], dtype=np.float64)
    fps = float(motion["fps"]) if "fps" in motion else 30.0
    doc, binary = read_glb(Path(args.glb))
    mapping = _bone_map(doc)
    if not mapping:
        print("GLB 里没有识别到人形骨架")
        return 2

    inverse = {node: smplx for smplx, node in mapping.items()}
    rest_joints = _smplx_rest_joints()
    children: dict[int, list[int]] = {}
    for child_index, parent_index in _smplx_parents().items():
        if parent_index >= 0:
            children.setdefault(parent_index, []).append(child_index)
    parents = {index: p for index, p in {
        i: (n.get("children") or [None]) and i for i, n in enumerate(doc.get("nodes", []))
    }.items()}

    # SMPL-X 侧：关节 → 父关节
    smplx_parent = _smplx_parents()

    wanted = [int(v) for v in args.frames.split(",") if v.strip()]

    # --self：不依赖 SMPL-X 模板，直接验证「角色的世界朝向 == rest 世界朝向 × SMPL-X 世界增量」
    if args.self and _is_mannequin(doc):
        # 自带骨架（mannequin）走的是“直接写关节平移”，所以校验的是位置而不是朝向。
        worst = 0.0
        for frame in wanted:
            state = sample_animation(doc, binary, frame, fps)
            positions = world_positions(doc, state)
            for node, smplx_index in inverse.items():
                error = float(np.linalg.norm(positions[node] - joints[min(frame, len(joints) - 1),
                                                                smplx_index]))
                worst = max(worst, error)
        print(f"自洽性检查（{Path(args.glb).name}）: 关节位置最大残差 {worst:.2e} m")
        print("PASS" if worst < 1e-3 else "FAIL")
        return 0 if worst < 1e-3 else 1

    if args.self:
        params = {k: motion[k] for k in ("body_pose", "global_orient", "left_hand_pose",
                                         "right_hand_pose", "betas") if k in motion}
        quats = local_rotations(params)
        deltas = _world_deltas(quats)
        rest_world = _rest_world(doc)
        parent_node = {}
        for index, node in enumerate(doc.get("nodes", [])):
            for child in node.get("children", []) or []:
                parent_node[int(child)] = index
        worst = 0.0
        for frame in wanted:
            state = sample_animation(doc, binary, frame, fps)
            world = _world_rotations(doc, state, parent_node)
            for node, smplx_index in inverse.items():
                expected = rest_world[node][0] @ deltas[smplx_index][min(frame, quats.shape[0] - 1)]
                delta = world[node] @ np.asarray(expected, dtype=np.float64).T
                angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(delta) - 1) / 2))))
                worst = max(worst, angle)
        print(f"自洽性检查（{Path(args.glb).name}）: 最大世界朝向残差 {worst:.2e}°")
        print("PASS" if worst < 1e-3 else "FAIL")
        return 0 if worst < 1e-3 else 1

    report = []
    for frame in wanted:
        if frame >= len(joints):
            continue
        state = sample_animation(doc, binary, frame, fps)
        world = world_positions(doc, state)
        rows = []
        for node, smplx_index in inverse.items():
            kids = [c for c in children.get(smplx_index, []) if c in mapping]
            if not kids:
                continue
            child_smplx = max(kids, key=lambda c: float(np.linalg.norm(rest_joints[c] - rest_joints[smplx_index])))
            child_node = mapping[child_smplx]
            target_dir = world[node] - world[child_node]
            source_dir = joints[frame, smplx_index] - joints[frame, child_smplx]
            if np.linalg.norm(target_dir) < 1e-8 or np.linalg.norm(source_dir) < 1e-8:
                continue
            cos = float(np.dot(target_dir, source_dir) /
                        (np.linalg.norm(target_dir) * np.linalg.norm(source_dir)))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            rows.append((SMPLX_NAMES[smplx_index], angle))
        report.append((frame, rows))

    print(f"=== {Path(args.glb).name} 与 {Path(args.motion).name} 的骨骼方向误差（越小越好）")
    headline = {}
    for frame, rows in report:
        if not rows:
            continue
        values = np.array([a for _, a in rows])
        print(f"帧 {frame:3d}: 平均 {values.mean():5.1f}°  中位 {np.median(values):5.1f}°  最大 {values.max():5.1f}°")
        for name, angle in sorted(rows, key=lambda r: -r[1])[:3]:
            print(f"        最差 {name:14s} {angle:5.1f}°")
        for name, angle in rows:
            headline.setdefault(name, []).append(angle)

    print("\n=== 分层平均（全部采样帧）")
    for name in sorted(headline, key=lambda n: -float(np.mean(headline[n])),):
        print(f"  {name:16s} {float(np.mean(headline[name])):5.1f}°")
    overall = float(np.mean([a for rows in headline.values() for a in rows])) if headline else 0.0
    print(f"\n总体平均: {overall:.2f}°")
    if args.json:
        Path(args.json).write_text(json.dumps({"overall": overall,
                                              "per_bone": {k: float(np.mean(v)) for k, v in headline.items()}},
                                             indent=2), encoding="utf-8")
    return 0


def _is_mannequin(doc: dict) -> bool:
    """自带骨架的节点名就是 SMPL-X 关节名（character 模式不会这样命名）。"""
    from zmocap.nodes.retarget import SMPLX_NAMES

    names = {node.get("name") for node in doc.get("nodes", [])}
    hits = sum(1 for name in SMPLX_NAMES if name in names)
    return hits >= 20


def _world_rotations(doc: dict, state: dict, parent_node: dict) -> dict:
    cache: dict[int, np.ndarray] = {}

    def resolve(index: int) -> np.ndarray:
        if index in cache:
            return cache[index]
        _translation, rotation, _scale = state[index]
        local = quat_to_mat(np.asarray(rotation, dtype=np.float64))
        cache[index] = resolve(parent_node[index]) @ local if index in parent_node else local
        return cache[index]

    for index in range(len(doc.get("nodes", []))):
        resolve(index)
    return cache


def _smplx_parents() -> dict:
    # SMPL-X 55 关节父索引
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
               18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
               35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
               21, 52, 53]
    return {i: p for i, p in enumerate(parents)}


if __name__ == "__main__":
    raise SystemExit(main())
