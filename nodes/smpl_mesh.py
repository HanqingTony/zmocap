"""SMPL-X 自带网格预览：参数 → 蒙皮 GLB，**零重定向**。

输出的模型和动作来自同一套 SMPL-X 参数，骨骼拓扑、蒙皮权重都来自官方模型文件，
因此"角色与动作是否一致"这个问题从根上消失——这正是用来做对照的标准答案。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_BETAS = "betas"
_SMPL_JOINTS = 55


def _model_path() -> Path:
    import os

    candidates = []
    env = os.environ.get("ZMOCAP_SMPLX")
    if env:
        candidates.append(Path(env))
    try:
        import folder_paths

        for root in folder_paths.get_folder_paths("zmocap") + folder_paths.get_folder_paths("zmocap_body_models"):
            candidates.append(Path(root) / "smplx" / "SMPLX_NEUTRAL.npz")
            candidates.append(Path(root) / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz")
    except Exception:
        pass
    candidates += [
        Path.home() / "zext/comfyui/models/zmocap/body_models/smplx/SMPLX_NEUTRAL.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("找不到 SMPL-X 模型（SMPLX_NEUTRAL.npz），设置 ZMOCAP_SMPLX 环境变量指向它")


class _SMPLX:
    """只加载一次官方模型数据。"""

    _cache: "_SMPLX | None" = None

    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        self.v_template = np.asarray(data["v_template"], dtype=np.float64)
        self.faces = np.asarray(data["f"], dtype=np.int32)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        self.shapedirs = np.asarray(data["shapedirs"], dtype=np.float64)[:, :, :300]
        self.j_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        self.parents = np.asarray(data["kintree_table"][0], dtype=np.int64)
        self.uv = np.asarray(data["vt"], dtype=np.float64) if "vt" in data.files else None
        self.uv_faces = np.asarray(data["ft"], dtype=np.int32) if "ft" in data.files else None

    @classmethod
    def load(cls) -> "_SMPLX":
        if cls._cache is None:
            cls._cache = cls(_model_path())
        return cls._cache

    def rest(self, betas: np.ndarray):
        shape = np.zeros_like(self.v_template)
        for index, value in enumerate(np.asarray(betas, dtype=np.float64)[:300]):
            shape += self.shapedirs[:, :, index] * value
        vertices = self.v_template + shape
        joints = self.j_regressor @ vertices
        return vertices, joints



def _add_accessor(appender, array, component, type_, count_last=None, minmax=False):
    """按指定 componentType 追加 accessor（add_floats 只支持 float32）。"""
    array = np.asarray(array)
    view = appender.add_view(array.tobytes())
    if len(array.shape) == 1:
        count, width = array.shape[0], 1
    else:
        count, width = array.shape[0], int(np.prod(array.shape[1:]))
    accessor = {"bufferView": view, "componentType": component, "count": int(count)}
    accessor["type"] = type_ if count_last is None else type_
    if minmax:
        flat = array.reshape(count, width)
        accessor["min"] = flat.min(axis=0).tolist()
        accessor["max"] = flat.max(axis=0).tolist()
    appender.doc["accessors"].append(accessor)
    return len(appender.doc["accessors"]) - 1


def _axis_angle_to_matrix(vectors: np.ndarray) -> np.ndarray:
    """(..., 3) 轴角 → (..., 3, 3) 旋转矩阵。"""
    v = np.asarray(vectors, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.divide(v, theta, out=np.zeros_like(v), where=theta > 1e-12)
    k = np.zeros(v.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -axis[..., 2], axis[..., 1]
    k[..., 1, 0], k[..., 1, 2] = axis[..., 2], -axis[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -axis[..., 1], axis[..., 0]
    sin = np.sin(theta)[..., None]
    cos = np.cos(theta)[..., None]
    eye = np.broadcast_to(np.eye(3), k.shape)
    return eye + sin * k + (1 - cos) * (k @ k)


def _local_rotations(params: dict, frames: int) -> np.ndarray:
    """每帧每关节的局部旋转 (frames, 55, 3, 3)。"""
    rotations = np.tile(np.eye(3), (frames, _SMPL_JOINTS, 1, 1))
    rotations[:, 0] = _axis_angle_to_matrix(np.asarray(params["global_orient"], dtype=np.float64))
    body = np.asarray(params["body_pose"], dtype=np.float64).reshape(frames, 21, 3)
    rotations[:, 1:22] = _axis_angle_to_matrix(body)
    for side, offset in (("left_hand_pose", 25), ("right_hand_pose", 40)):
        if side in params:
            hand = np.asarray(params[side], dtype=np.float64).reshape(frames, 15, 3)
            rotations[:, offset:offset + 15] = _axis_angle_to_matrix(hand)
    return rotations


def _quat_xyzw(matrix: np.ndarray) -> list[float]:
    from .sticks import _quat_xyzw as convert

    return convert(matrix)


def write_smpl_glb(motion: dict, out_path: Path, sticks: bool = True) -> tuple[Path, dict]:
    from .retarget import _Appender

    model = _SMPLX.load()
    params = motion["params"]
    frames = int(np.asarray(params["global_orient"]).shape[0])
    fps = float(motion.get("fps", 30.0))
    betas = np.asarray(params.get(_BETAS, np.zeros(10)), dtype=np.float64)
    if betas.ndim == 2:
        betas = betas[0]

    vertices, joints = model.rest(betas)
    rotations = _local_rotations(params, frames)
    transl = np.asarray(params.get("transl", np.zeros((frames, 3))), dtype=np.float64)

    parents = model.parents
    doc: dict = {"asset": {"version": "2.0", "generator": "zmocap-smplx"}}
    appender = _Appender(doc, bytearray())

    # ── 网格 ─────────────────────────────────────────────────────────────
    normals = np.zeros_like(vertices)
    tri = vertices[model.faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, model.faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    top = np.argsort(-model.weights, axis=1)[:, :4]
    skin_weights = np.take_along_axis(model.weights, top, axis=1)
    skin_weights /= np.maximum(skin_weights.sum(axis=1, keepdims=True), 1e-9)
    joints_attr = top.astype(np.uint16)
    weights_attr = skin_weights.astype(np.float32)

    attributes = {
        "POSITION": appender.add_floats(vertices.astype(np.float32), "VEC3", minmax=True),
        "NORMAL": appender.add_floats(normals.astype(np.float32), "VEC3"),
        "JOINTS_0": _add_accessor(appender, joints_attr, 5123, "VEC4"),
        "WEIGHTS_0": appender.add_floats(weights_attr, "VEC4"),
    }
    # SMPL-X 的 vt 是独立 UV 布局（11313 条），和顶点数（10475）不一致；
    # 直接当 VEC2 塞进逐顶点 primitive 会让 accessor count 不匹配（glTF 非法，
    # three.js 可能拒绝/错位）。材质没有贴图，不写 UV 即可。
    if model.uv is not None and model.uv.shape[0] == vertices.shape[0]:
        attributes["TEXCOORD_0"] = appender.add_floats(model.uv.astype(np.float32), "VEC2")
    # faces 是 (F,3)；索引是 SCALAR，必须按 F*3 写，不能按 F 写。
    index_accessor = _add_accessor(appender, model.faces.astype(np.uint32).reshape(-1), 5125, "SCALAR")
    doc["materials"] = [{
        "name": "zmocap_skin",
        "pbrMetallicRoughness": {"baseColorFactor": [0.85, 0.72, 0.62, 1.0],
                                 "metallicFactor": 0.0, "roughnessFactor": 0.75},
        "doubleSided": False,
    }]
    doc["meshes"] = [{
        "name": "SMPLX_body",
        "primitives": [{"attributes": attributes, "indices": index_accessor, "material": 0, "mode": 4}],
    }]

    # ── 骨骼 ─────────────────────────────────────────────────────────────
    nodes = []
    for index in range(_SMPL_JOINTS):
        parent = int(parents[index])
        local = joints[index] if parent < 0 or parent >= _SMPL_JOINTS else joints[index] - joints[parent]
        nodes.append({
            "name": f"smplx_{index:02d}",
            "translation": [float(v) for v in local],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "children": [j for j in range(_SMPL_JOINTS) if int(parents[j]) == index],
        })
    doc["nodes"] = nodes
    doc["scene"] = 0

    # 逆绑定矩阵（rest 只有平移）
    inverse = np.tile(np.eye(4), (_SMPL_JOINTS, 1, 1))
    inverse[:, :3, 3] = -joints
    # glTF 的 MAT4 是列主序：先整体转置再平铺
    ibm_view = appender.add_floats(inverse.transpose(0, 2, 1).reshape(_SMPL_JOINTS, 16).astype(np.float32), "MAT4")
    doc["skins"] = [{
        "name": "SMPLX_skin",
        "joints": list(range(_SMPL_JOINTS)),
        "inverseBindMatrices": len(doc["accessors"]) - 1,
        "skeleton": 0,
    }]
    # 网格单独一个节点（不与根骨骼复用）：three.js 等对"网格节点 = 关节节点"处理不一致
    nodes.append({"name": "SMPLX_mesh", "mesh": 0, "skin": 0})
    doc["scenes"] = [{"nodes": [0, len(nodes) - 1]}]
    del ibm_view

    # ── 动画 ─────────────────────────────────────────────────────────────
    times = np.arange(frames, dtype=np.float64) / fps
    time_accessor = appender.add_floats(times.astype(np.float32), "SCALAR", minmax=True)
    samplers, channels = [], []
    for index in range(_SMPL_JOINTS):
        values = np.stack([_quat_xyzw(r) for r in rotations[:, index]], axis=0).astype(np.float32)
        samplers.append({"input": time_accessor, "interpolation": "LINEAR",
                         "output": appender.add_floats(values, "VEC4")})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": index, "path": "rotation"}})
    if np.abs(transl).max() > 1e-9:
        # 根节点 rest 平移 = 骨盆静止位置，动画平移要在它之上叠加
        root_translation = joints[0] + transl
        samplers.append({"input": time_accessor, "interpolation": "LINEAR",
                         "output": appender.add_floats(root_translation.astype(np.float32), "VEC3")})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": 0, "path": "translation"}})
    doc["animations"] = [{"name": "zmocap", "samplers": samplers, "channels": channels}]

    stick_count = 0
    if sticks:
        from .sticks import add_stick_layer

        parents_map = {i: int(parents[i]) for i in range(_SMPL_JOINTS) if 0 <= int(parents[i]) < _SMPL_JOINTS}
        by_name = {node["name"]: i for i, node in enumerate(nodes)}
        try:
            stick_count = add_stick_layer(appender, doc, [n["name"] for n in nodes], parents_map, by_name)
        except Exception as exc:                    # 小棍只是显示层
            print(f"[zmocap] SMPL 小棍失败（{exc}）")

    appender.finish()
    out_path = Path(out_path)
    out_path.write_bytes(_write(doc, appender.binary))
    info = {
        "mode": "smplx",
        "bones": _SMPL_JOINTS,
        "frames": frames,
        "vertices": int(vertices.shape[0]),
        "sticks": stick_count,
        "height": round(float(vertices[:, 1].max() - vertices[:, 1].min()), 3),
    }
    print(f"[zmocap] SMPL-X 网格 {info['vertices']} 顶点 / {_SMPL_JOINTS} 骨骼 / {frames} 帧")
    return out_path, info


def _write(doc: dict, binary: bytes) -> bytes:
    json_bytes = json.dumps(doc, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    header = b"glTF" + struct.pack("<II", 2, total)
    return (header
            + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
