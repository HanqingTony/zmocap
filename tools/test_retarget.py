"""重定向数学的单元测试：不依赖任何真实角色或动作数据。

构造一条 6 截骨链（pelvis → spine1 → spine2 → spine3 → neck → head），给角色骨骼
任意的 rest 旋转，再给 SMPL-X 一组已知的局部旋转，然后用 GLB 动画做前向运动学，
把角色的世界朝向与「rest 世界朝向 × SMPL-X 世界增量」对比。

这里误差不为 0 就说明 `_add_animation` 的相似变换写错了；为 0 则真实角色上的偏差
只可能来自骨骼命名/朝向约定（映射问题），与数学无关。

运行::

    python -m zmocap.tools.test_retarget
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

CHAIN = [
    (0, "hips"),
    (3, "spine"),
    (6, "chest"),
    (9, "upperchest"),
    (12, "neck"),
    (15, "head"),
]


def mat_from_axis(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def quat_from_mat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    out = np.zeros(4)
    out[i] = 0.25 * s
    out[j] = (m[j, i] + m[i, j]) / s
    out[k] = (m[k, i] + m[i, k]) / s
    out[3] = (m[k, j] - m[j, k]) / s
    return out


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from zmocap.nodes import retarget
    from zmocap.tools import fk_check

    frames, fps = 5, 30.0
    rng = np.random.default_rng(7)

    # 角色骨骼：任意 rest 局部旋转 + 任意 rest 平移（故意让朝向千奇百怪）
    nodes, parent_of = [], {}
    for position, (_smplx, name) in enumerate(CHAIN):
        node = {"name": name,
                "translation": [0.0, 0.1 * (position + 1), 0.0],
                "rotation": quat_from_mat(mat_from_axis(rng.normal(size=3),
                                                        rng.uniform(-70, 70))).tolist()}
        if position:
            node["children"] = []
            nodes[position - 1]["children"] = nodes[position - 1].get("children", []) + [position]
            parent_of[position] = position - 1
        nodes.append(node)
    doc = {"asset": {"version": "2.0"}, "nodes": nodes, "scenes": [{"nodes": [0]}]}
    mapping = {smplx_index: position for position, (smplx_index, _name) in enumerate(CHAIN)}

    # SMPL-X 局部旋转：spine1 / neck / head 有动作，其余保持 T-pose
    quats = np.zeros((frames, len(retarget.SMPLX_PARENTS), 4), dtype=np.float64)
    quats[..., 3] = 1.0
    moving = {3: mat_from_axis([1, 0, 0], 32.0),
              12: mat_from_axis([0, 0, 1], 21.0),
              15: mat_from_axis([0, 1, 0], 14.0)}
    for frame in range(frames):
        weight = frame / max(1, frames - 1)
        for smplx_index, matrix in moving.items():
            blended = (1 - weight) * np.eye(3) + weight * matrix
            quats[frame, smplx_index, :] = quat_from_mat(blended)

    appender = retarget._Appender(doc, bytearray())
    assert retarget._add_animation(appender, doc, mapping, quats, fps), "没有生成动画"
    appender.finish()

    deltas = retarget._world_deltas(quats)
    rest_world = retarget._rest_world(doc)
    worst, worst_label = 0.0, ""
    for frame in range(frames):
        state = fk_check.sample_animation(doc, bytes(appender.binary), frame, fps)
        current = _world_rotations(doc, state, parent_of)
        for smplx_index, position in mapping.items():
            expected = rest_world[position][0] @ deltas[smplx_index][frame]
            delta = current[position] @ expected.T
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(delta) - 1) / 2))))
            if angle > worst:
                worst, worst_label = angle, f"帧 {frame} {CHAIN[position][1]}"
    print(f"最大世界朝向误差: {worst:.3e}°  ({worst_label or '全部通过'})")
    ok = worst < 1e-5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _world_rotations(doc: dict, state: dict, parent_of: dict) -> dict:
    cache: dict[int, np.ndarray] = {}

    def resolve(index: int) -> np.ndarray:
        if index in cache:
            return cache[index]
        _translation, rotation, _scale = state[index]
        from zmocap.nodes.retarget import _mat_from_quat

        local = _mat_from_quat(np.asarray(rotation, dtype=np.float64))
        cache[index] = resolve(parent_of[index]) @ local if index in parent_of else local
        return cache[index]

    for index in range(len(doc["nodes"])):
        resolve(index)
    return cache


if __name__ == "__main__":
    raise SystemExit(main())
