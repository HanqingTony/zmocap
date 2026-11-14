"""导出节点：NPZ + BVH + contacts JSON/NPZ。"""

import json
from pathlib import Path

import numpy as np


class ZMocapExport:
    """把动作序列写成 NPZ（关节 + SMPL-X 参数）和 BVH（骨架动画）。"""

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_SMPLX",),
                "filename": ("STRING", {"default": "zmocap_motion"}),
            },
            "optional": {
                "write_bvh": ("BOOLEAN", {"default": True}),
                "contacts": ("ZMOCAP_CONTACTS",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, motion, filename, write_bvh=True, contacts=None):
        from ..motion import validate_clip

        validate_clip(motion, "smplx")
        try:
            import folder_paths

            outdir = Path(folder_paths.get_output_directory())
        except Exception:
            outdir = Path("output")
        outdir.mkdir(parents=True, exist_ok=True)
        joints = np.asarray(motion["joints"], dtype=np.float32)
        npz_path = outdir / f"{filename}.npz"
        payload = {k: _array(v) for k, v in motion["params"].items()}
        # 保留 contact 所需的原始证据，便于重新检测/校验。
        if motion.get("static_conf_logits") is not None:
            payload["static_conf_logits"] = _array(motion["static_conf_logits"])
        if contacts is not None:
            for source, target in (("contact", "contact"),
                                   ("probability", "contact_probability"),
                                   ("height", "contact_height")):
                value = (contacts.get("arrays") or {}).get(source)
                if value is not None:
                    payload[target] = _array(value)
        np.savez_compressed(npz_path, joints=joints, fps=motion.get("fps", 30.0), **payload)
        paths = [str(npz_path)]
        if write_bvh:
            bvh_path = outdir / f"{filename}.bvh"
            text = _bvh(motion)
            bvh_path.write_text(text, encoding="utf-8")
            paths.append(str(bvh_path))
            check = _verify_bvh(text, motion)
            print(
                "[zmocap] BVH 回读 "
                f"{check['joints']} 关节 / {check['frames']} 帧 / 最大误差 {check['max_error'] * 1000:.2f} mm"
            )
            if not check["ok"]:
                print("[zmocap] 警告：BVH 回读误差偏大，检查 rest 骨架或根位移")
        if contacts is not None:
            paths.extend(_write_contacts(outdir, filename, motion, contacts))
        return ("\n".join(paths),)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_contacts(outdir: Path, filename: str, motion: dict, contacts: dict):
    """JSON 存区间/元数据，NPZ 存逐帧数组；两者用同名配对。"""
    document = {key: value for key, value in contacts.items() if key != "arrays"}
    document = _jsonable(document)
    json_path = outdir / f"{filename}.contacts.json"
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    arrays = {
        key: _array(value)
        for key, value in (contacts.get("arrays") or {}).items()
        if value is not None
    }
    npz_path = outdir / f"{filename}.contacts.npz"
    np.savez_compressed(
        npz_path,
        fps=float(contacts.get("fps", motion.get("fps", 30.0))),
        frames=int(contacts.get("frames", len(np.asarray(motion["joints"])))),
        **arrays,
    )
    return [str(json_path), str(npz_path)]


class BvhError(ValueError):
    """BVH 导出无法表达输入（例如缺 rest 骨架）。"""


def _rest_skeleton(motion):
    """取出写 BVH 需要的 rest 骨架：offsets + parents。

    骨架与参数同源（SMPL-X 中立模型），不接受"用第 0 帧充当 rest"这种做法：
    第 0 帧是动作姿态，拿它当 rest 会让 FK 结果偏移，且与通道语义不符。
    """
    params = motion.get("params") or {}
    if not params:
        raise BvhError("BVH 导出需要 SMPL-X 参数（params 缺失）；VRM/humanoid 运动请先导出 GLB")
    import numpy as _np

    from .smpl_mesh import _SMPLX

    betas = _np.asarray(params.get("betas", _np.zeros(10)), dtype=_np.float64)
    if betas.ndim == 2:
        betas = betas[0]
    model = _SMPLX.load()
    _, rest = model.rest(betas)
    parents = [int(parent) for parent in model.parents]
    parents[0] = -1
    names = [f"J{index:02d}" for index in range(rest.shape[0])]
    return rest, parents, names


def _euler_zxy(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → BVH 通道顺序 (Z, X, Y) 的欧拉角。

    BVH 的旋转按通道列出顺序依次相乘，`Zrotation Xrotation Yrotation` 即 ``R = Rz · Rx · Ry``。
    """
    import numpy as _np

    sin_x = _np.clip(matrix[..., 2, 1], -1.0, 1.0)
    x = _np.arcsin(sin_x)
    cos_x = _np.cos(x)
    safe = _np.abs(cos_x) > 1e-6
    y = _np.where(safe, _np.arctan2(-matrix[..., 2, 0], matrix[..., 2, 2]), 0.0)
    z = _np.where(safe, _np.arctan2(-matrix[..., 0, 1], matrix[..., 1, 1]), _np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))
    return _np.stack([z, x, y], axis=-1)


def _bvh(motion) -> str:
    """按 SMPL-X rest 骨架写合法 BVH：真实 offsets、完整通道、真实局部旋转与根位移。"""
    import numpy as _np

    from .retarget import _mat_from_quat, local_rotations

    joints = _np.asarray(motion["joints"], dtype=_np.float64)
    params = motion.get("params") or {}
    fps = float(motion.get("fps", 30.0))
    frames = joints.shape[0]

    rest, parents, names = _rest_skeleton(motion)
    offsets = _np.zeros_like(rest)
    for index, parent in enumerate(parents):
        if parent >= 0:
            offsets[index] = rest[index] - rest[parent]

    quats = local_rotations(params)                     # (T, 55, 4) xyzw
    matrices = _mat_from_quat(quats)
    angles = _euler_zxy(matrices)

    root_translation = _np.asarray(params.get("transl", _np.zeros((frames, 3))), dtype=_np.float64)
    if root_translation.shape[0] != frames:
        root_translation = rest[0][None, :] + _np.zeros((frames, 3))
    else:
        root_translation = rest[0][None, :] + root_translation  # transl 是相对中立骨盆的位置

    children = {index: [] for index in range(len(names))}
    for index, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(index)

    lines = ["HIERARCHY", f"ROOT {names[0]}", "{"]
    lines.append("  OFFSET 0.000000 0.000000 0.000000")
    lines.append("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")

    emit_order = [0]

    def emit(index, depth):
        pad = "  " * depth
        for child in children[index]:
            offset = offsets[child]
            lines.append(f"{pad}JOINT {names[child]}")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
            emit_order.append(child)
            emit(child, depth + 1)
            lines.append(f"{pad}}}")

    emit(0, 1)
    lines.append("}")
    lines.append("MOTION")
    lines.append(f"Frames: {frames}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")
    for frame in range(frames):
        values = list(root_translation[frame]) + list(angles[frame, 0])
        # MOTION 的通道值必须按 HIERARCHY 的 DFS 声明顺序排列，而不是按关节 id
        for index in emit_order[1:]:
            values.extend(angles[frame, index])
        lines.append(" ".join(f"{value:.6f}" for value in values))
    return "\n".join(lines) + "\n"


def _quat_from_euler_zxy(euler) -> np.ndarray:
    """(Z, X, Y) 欧拉角 → 四元数，对应 BVH 的 ``R = Rz · Rx · Ry``。"""
    import numpy as _np

    from .retarget import _quat_from_mat

    z, x, y = (float(value) for value in euler)
    cz, sz = _np.cos(z), _np.sin(z)
    cx, sx = _np.cos(x), _np.sin(x)
    cy, sy = _np.cos(y), _np.sin(y)
    rz = _np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rx = _np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = _np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return _quat_from_mat(rz @ rx @ ry)


def _rotate(quat, vector) -> np.ndarray:
    """用四元数 (x, y, z, w) 旋转向量。"""
    import numpy as _np

    x, y, z, w = (float(value) for value in quat)
    u = _np.array([x, y, z], dtype=_np.float64)
    v = _np.asarray(vector, dtype=_np.float64)
    return v + 2.0 * _np.cross(u, _np.cross(u, v) + w * v)


def _parse_bvh(hierarchy: str):
    """从 HIERARCHY 段解析 offsets / 父索引 / 关节名。

    返回的顺序是**文件里的 DFS 顺序**（BVH 就是这么写的），需要按名字映射回关节 id 使用。
    """
    import numpy as _np

    offsets: list[list[float]] = []
    parents: list[int] = []
    names: list[str] = []
    stack: list[int] = []
    for raw in hierarchy.splitlines():
        line = raw.strip()
        if line.startswith(("ROOT", "JOINT")):
            names.append(line.split(maxsplit=1)[1])
            parents.append(stack[-1] if stack else -1)
        elif line.startswith("OFFSET"):
            offsets.append([float(token) for token in line.split()[1:4]])
        elif line == "{":
            stack.append(len(parents) - 1)
        elif line == "}":
            stack.pop()
    return _np.asarray(offsets, dtype=_np.float64), parents, names


def _verify_bvh(text: str, motion, tolerance: float = 5e-3) -> dict:
    """回读 BVH 并做三项校验：offsets 是否等于 rest、每帧角度是否等于 params、FK 是否还原姿态。"""
    import numpy as _np

    from .retarget import _mat_from_quat, local_rotations

    header, _, body = text.partition("MOTION")
    offsets_dfs, parents_dfs, names = _parse_bvh(header)
    count = len(names)

    def joint_id(name: str) -> int:
        tail = name.lstrip("Jj")
        return int(tail) if tail.isdigit() else names.index(name)

    position = {name: joint_id(name) for name in names}
    offsets = _np.zeros((count, 3))
    parents = [-1] * count
    for dfs_index in range(count):
        target = position[names[dfs_index]]
        offsets[target] = offsets_dfs[dfs_index]
        parent_dfs = parents_dfs[dfs_index]
        parents[target] = position[names[parent_dfs]] if parent_dfs >= 0 else -1

    rows = [row for row in body.strip().splitlines()[2:] if row.strip()]
    frames = len(rows)
    values = _np.array([[float(token) for token in row.split()] for row in rows])

    # 文件里的角度按 DFS 顺序排，先还原成 id 顺序
    angles = _np.zeros((frames, count, 3))
    angles[:, 0] = values[:, 3:6]
    for dfs_index in range(1, count):
        start_index = 6 + 3 * (dfs_index - 1)
        angles[:, position[names[dfs_index]]] = values[:, start_index:start_index + 3]

    rest, expected_parents, expected_names = _rest_skeleton(motion)
    offset_error = float(_np.abs(offsets - _np.asarray([
        rest[i] - rest[p] if p >= 0 else _np.zeros(3) for i, p in enumerate(expected_parents)
    ])).max())
    parent_ok = parents == expected_parents

    params = motion.get("params") or {}
    quats = local_rotations(params)
    matrices = _mat_from_quat(quats)
    expected_angles = _euler_zxy(matrices)
    # 欧拉角有 ±2π 等价性，比较 sin/cos
    angle_error = float(_np.abs(
        _np.sin(angles) - _np.sin(expected_angles[:frames])
    ).max())

    world = _np.zeros((frames, count, 3))
    world_rot = _np.zeros((frames, count, 4))
    for index in range(count):
        world_rot[:, index] = _np.stack([_quat_from_euler_zxy(row) for row in angles[:, index]])
        if index == 0:
            world[:, index] = values[:, 0:3]
        else:
            parent = parents[index]
            local = _np.stack([_rotate(world_rot[frame, parent], offsets[index]) for frame in range(frames)])
            world[:, index] = world[:, parent] + local

    translation = _np.asarray(params.get("transl", _np.zeros((frames, 3))), dtype=_np.float64)
    root = rest[0][None, :] + translation[:frames] if translation.shape[0] >= frames else _np.tile(rest[0], (frames, 1))
    reference = _np.zeros((frames, count, 3))
    reference[:, 0] = root
    for index in range(1, count):
        parent = expected_parents[index]
        reference[:, index] = reference[:, parent] + _np.einsum(
            "fij,j->fi", matrices[:frames, parent], rest[index] - rest[parent]
        )
    fk_error = float(_np.abs(world - reference).max())

    return {
        "frames": frames,
        "joints": count,
        "offset_error": offset_error,
        "parent_ok": bool(parent_ok),
        "angle_error": angle_error,
        "max_error": fk_error,
        "ok": bool(offset_error <= 1e-4 and parent_ok and angle_error <= 1e-4 and fk_error <= tolerance),
    }


