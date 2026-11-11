"""角色绑定预览：把 SMPL-X 动作写成带头部/手指骨骼的动画 GLB。

两种模式：

* `mannequin`：自带骨架（55 个球关节 + 关节连线），把每帧关节位置写成节点平移动画。
* `character`：读入用户角色（GLB/VRM 的 glTF 二进制），按骨骼名把 SMPL-X 的
  局部旋转写进角色骨架的 rotation 动画，角色自带骨骼/蒙皮不动。

输出统一是 GLB 文件，可直接接到 ComfyUI 官方 `Preview3D` 节点预览（前端 three.js
带 AnimationMixer，会播放动画）。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# SMPL-X 55 关节的父索引（与 smplx 包的 b.parents 完全一致）
SMPLX_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]

# SMPL-X 关节名（0-54），用于和常见角色骨架名对齐
SMPLX_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye", "right_eye",
    "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2",
    "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3",
]

# 目标骨骼名 → SMPL-X 关节索引。
#
# 关键：SMPL-X 的旋转是“关节坐标系的增量”，它驱动的是 *父关节→本关节这一段骨骼*；
# 而 VRM/Mixamo 的骨骼节点旋转驱动的是 *自身节点→子骨* 的骨段。因此映射必须
# 错开一级：角色的 UpperArm（肩膀→手肘段）对应 SMPL-X 的 left_shoulder(16)，
# 而不是 left_elbow(18)；UpperLeg 对应 left_hip(1) 而不是 left_knee(4)。
# 按序号顺移一格是这里最容易错的地方。
_BONE_TABLE: list[tuple[int, tuple[str, ...]]] = [
    (0, ("hips", "mixamorighips", "j_bip_c_hips", "pelvis")),
    (3, ("spine", "mixamorigspine", "j_bip_c_spine")),
    (6, ("chest", "spine1", "mixamorigspine1", "j_bip_c_chest")),
    (9, ("upperchest", "spine2", "mixamorigspine2", "j_bip_c_upperchest")),
    (12, ("neck", "mixamorigneck", "j_bip_c_neck")),
    (15, ("head", "mixamorighead", "j_bip_c_head")),
    # 左臂：collar → shoulder 段 / shoulder → elbow 段 / elbow → wrist 段 / 手
    (13, ("leftshoulder", "mixamorigleftshoulder", "j_bip_l_shoulder", "leftcollar")),
    (16, ("leftarm", "mixamorigleftarm", "j_bip_l_upperarm", "leftupperarm")),
    (18, ("leftforearm", "mixamorigleftforearm", "j_bip_l_lowerarm", "leftlowerarm")),
    (20, ("lefthand", "mixamoriglefthand", "j_bip_l_hand", "lefthandjnt")),
    # 右臂
    (14, ("rightshoulder", "mixamorigrightshoulder", "j_bip_r_shoulder", "rightcollar")),
    (17, ("rightarm", "mixamorigrightarm", "j_bip_r_upperarm", "rightupperarm")),
    (19, ("rightforearm", "mixamorigrightforearm", "j_bip_r_lowerarm", "rightlowerarm")),
    (21, ("righthand", "mixamorigrighthand", "j_bip_r_hand", "righthandjnt")),
    # 左腿：hip → knee 段 / knee → ankle 段 / ankle → toe 段 / 脚趾
    (1, ("leftupleg", "mixamorigleftupleg", "j_bip_l_upperleg", "leftupperleg", "lefthip")),
    (4, ("leftleg", "mixamorigleftleg", "j_bip_l_lowerleg", "leftlowerleg", "leftknee")),
    (7, ("leftfoot", "mixamorigleftfoot", "j_bip_l_foot", "leftankle")),
    (10, ("j_bip_l_toebase", "lefttoebase", "mixamoriglefttoebase", "j_bip_l_toes", "lefttoe")),
    (2, ("rightupleg", "mixamorigrightupleg", "j_bip_r_upperleg", "rightupperleg", "righthip")),
    (5, ("rightleg", "mixamorigrightleg", "j_bip_r_lowerleg", "rightlowerleg", "rightknee")),
    (8, ("rightfoot", "mixamorigrightfoot", "j_bip_r_foot", "rightankle")),
    (11, ("j_bip_r_toebase", "righttoebase", "mixamorigrighttoebase", "j_bip_r_toes", "righttoe")),
]

# 手指（VRM 命名，节数比 SMPL-X 一一对应）
_FINGER_TABLE: list[tuple[int, tuple[str, ...]]] = [
    (25, ("j_bip_l_index1", "leftindex1", "mixamoriglefthandindex1")),
    (26, ("j_bip_l_index2", "leftindex2", "mixamoriglefthandindex2")),
    (27, ("j_bip_l_index3", "leftindex3", "mixamoriglefthandindex3")),
    (28, ("j_bip_l_middle1", "leftmiddle1", "mixamoriglefthandmiddle1")),
    (29, ("j_bip_l_middle2", "leftmiddle2", "mixamoriglefthandmiddle2")),
    (30, ("j_bip_l_middle3", "leftmiddle3", "mixamoriglefthandmiddle3")),
    (31, ("j_bip_l_little1", "leftpinky1", "mixamoriglefthandpinky1")),
    (32, ("j_bip_l_little2", "leftpinky2", "mixamoriglefthandpinky2")),
    (33, ("j_bip_l_little3", "leftpinky3", "mixamoriglefthandpinky3")),
    (34, ("j_bip_l_ring1", "leftring1", "mixamoriglefthandring1")),
    (35, ("j_bip_l_ring2", "leftring2", "mixamoriglefthandring2")),
    (36, ("j_bip_l_ring3", "leftring3", "mixamoriglefthandring3")),
    (37, ("j_bip_l_thumb1", "leftthumb1", "mixamoriglefthandthumb1")),
    (38, ("j_bip_l_thumb2", "leftthumb2", "mixamoriglefthandthumb2")),
    (39, ("j_bip_l_thumb3", "leftthumb3", "mixamoriglefthandthumb3")),
    (40, ("j_bip_r_index1", "rightindex1", "mixamorigrighthandindex1")),
    (41, ("j_bip_r_index2", "rightindex2", "mixamorigrighthandindex2")),
    (42, ("j_bip_r_index3", "rightindex3", "mixamorigrighthandindex3")),
    (43, ("j_bip_r_middle1", "rightmiddle1", "mixamorigrighthandmiddle1")),
    (44, ("j_bip_r_middle2", "rightmiddle2", "mixamorigrighthandmiddle2")),
    (45, ("j_bip_r_middle3", "rightmiddle3", "mixamorigrighthandmiddle3")),
    (46, ("j_bip_r_little1", "rightpinky1", "mixamorigrighthandpinky1")),
    (47, ("j_bip_r_little2", "rightpinky2", "mixamorigrighthandpinky2")),
    (48, ("j_bip_r_little3", "rightpinky3", "mixamorigrighthandpinky3")),
    (49, ("j_bip_r_ring1", "rightring1", "mixamorigrighthandring1")),
    (50, ("j_bip_r_ring2", "rightring2", "mixamorigrighthandring2")),
    (51, ("j_bip_r_ring3", "rightring3", "mixamorigrighthandring3")),
    (52, ("j_bip_r_thumb1", "rightthumb1", "mixamorigrighthandthumb1")),
    (53, ("j_bip_r_thumb2", "rightthumb2", "mixamorigrighthandthumb2")),
    (54, ("j_bip_r_thumb3", "rightthumb3", "mixamorigrighthandthumb3")),
]

# 兼容旧名
_ALIASES = {SMPLX_NAMES[index]: list(aliases) for index, aliases in _BONE_TABLE}

_COMPONENT_FLOAT = 5126


def _normalize(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace(".", "")


def _axis_angle_to_quat(aa) -> np.ndarray:
    """(..., 3) 轴角 → (..., 4) 四元数 (x, y, z, w)。"""
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    angle = np.linalg.norm(aa, axis=1, keepdims=True)
    axis = np.divide(aa, angle, out=np.zeros_like(aa), where=angle > 1e-9)
    half = angle / 2.0
    sin_half = np.sin(half)
    return np.concatenate([axis * sin_half, np.cos(half)], axis=1)


def local_rotations(params) -> np.ndarray:
    """SMPL-X 参数 → (T, 55, 4) 局部四元数，顺序与 SMPLX_PARENTS 一致。"""
    import torch

    def arr(value, width):
        if value is None:
            return None
        value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return value.reshape(value.shape[0], width)

    count = int(params["betas"].shape[0])
    body = arr(params.get("body_pose"), 63)
    glob = arr(params.get("global_orient"), 3)
    left = arr(params.get("left_hand_pose"), 45)
    right = arr(params.get("right_hand_pose"), 45)

    rotations = np.zeros((count, 55, 3), dtype=np.float64)
    if glob is not None:
        rotations[:, 0] = glob
    if body is not None:
        rotations[:, 1:22] = body.reshape(count, 21, 3)
    if left is not None:
        rotations[:, 25:40] = left.reshape(count, 15, 3)
    if right is not None:
        rotations[:, 40:55] = right.reshape(count, 15, 3)
    return _axis_angle_to_quat(rotations).reshape(count, 55, 4)


# --- 极简 GLB 读写 ---------------------------------------------------------
def _read_glb(path: Path):
    raw = Path(path).read_bytes()
    magic, version, _length = struct.unpack("<III", raw[:12])
    if magic != 0x46546C67:
        raise ValueError(f"不是 GLB 文件：{path}")
    offset, doc, binary = 12, None, b""
    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack("<II", raw[offset:offset + 8])
        chunk = raw[offset + 8:offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += 8 + chunk_len + ((4 - chunk_len % 4) % 4 if chunk_len % 4 else 0)
    if doc is None:
        raise ValueError(f"GLB 缺少 JSON 段：{path}")
    return doc, bytearray(binary)


def _write_glb(path: Path, doc: dict, binary: bytearray) -> Path:
    json_chunk = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    bin_chunk = bytes(binary)
    bin_chunk += b"\x00" * ((4 - len(bin_chunk) % 4) % 4)
    total = 12 + 8 + len(json_chunk) + (8 + len(bin_chunk) if bin_chunk else 0)
    out = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    if bin_chunk:
        out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


class _Appender:
    """往已有 glTF 文档里追加 accessor/bufferView 的助手。"""

    def __init__(self, doc: dict, binary: bytearray):
        self.doc = doc
        self.binary = binary
        self.doc.setdefault("accessors", [])
        self.doc.setdefault("bufferViews", [])
        self.doc.setdefault("buffers", [{"byteLength": len(binary)}])

    def _align(self, multiple: int = 4):
        pad = (multiple - len(self.binary) % multiple) % multiple
        if pad:
            self.binary.extend(b"\x00" * pad)

    def add_view(self, data: bytes) -> int:
        self._align()
        offset = len(self.binary)
        self.binary.extend(data)
        self.doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        return len(self.doc["bufferViews"]) - 1

    def add_floats(self, array, type_: str, minmax: bool = False) -> int:
        array = np.asarray(array, dtype=np.float32)
        view = self.add_view(array.tobytes())
        accessor = {
            "bufferView": view,
            "componentType": _COMPONENT_FLOAT,
            "count": int(array.shape[0]),
            "type": type_,
        }
        if minmax:
            accessor["min"] = array.min(axis=0).tolist()
            accessor["max"] = array.max(axis=0).tolist()
        self.doc["accessors"].append(accessor)
        return len(self.doc["accessors"]) - 1

    def finish(self):
        self.doc["buffers"] = [{"byteLength": len(self.binary)}]


def _bone_map(doc: dict) -> dict:
    """SMPL-X 关节索引 → 目标 GLB 的节点索引（一个节点只认一次）。"""
    lookup = {}
    for index, node in enumerate(doc.get("nodes", [])):
        name = node.get("name")
        if name:
            lookup.setdefault(_normalize(name), index)
    mapping: dict[int, int] = {}
    used: set[int] = set()

    # 精确同名优先：自带骨架的节点名就是 SMPL-X 关节名（pelvis / left_shoulder …），
    # 别名表里也有同名字符串，先按精确名对齐，避免 left_collar 抢走 left_shoulder。
    for smplx_index, name in enumerate(SMPLX_NAMES):
        node_index = lookup.get(_normalize(name))
        if node_index is not None and node_index not in used:
            mapping[smplx_index] = node_index
            used.add(node_index)

    for smplx_index, aliases in _BONE_TABLE + _FINGER_TABLE:
        if smplx_index in mapping:
            continue
        for alias in aliases:
            node_index = lookup.get(_normalize(alias))
            if node_index is not None and node_index not in used:
                mapping[smplx_index] = node_index
                used.add(node_index)
                break
    return mapping


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 乘积（批量），四元数顺序 (x, y, z, w)。"""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _mat_from_quat(q) -> np.ndarray:
    """四元数 (..., 4) → 旋转矩阵 (..., 3, 3)，顺序 (x, y, z, w)。"""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.divide(q, np.where(norm < 1e-12, 1.0, norm))
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[..., 0, 1] = 2.0 * (x * y - z * w)
    out[..., 0, 2] = 2.0 * (x * z + y * w)
    out[..., 1, 0] = 2.0 * (x * y + z * w)
    out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[..., 1, 2] = 2.0 * (y * z - x * w)
    out[..., 2, 0] = 2.0 * (x * z - y * w)
    out[..., 2, 1] = 2.0 * (y * z + x * w)
    out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    """旋转矩阵 (..., 3, 3) → 四元数 (..., 4)，顺序 (x, y, z, w)。"""
    m = np.asarray(m, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
    positive = trace > 0
    s = np.sqrt(np.where(positive, trace + 1.0, 1.0)) * 2.0
    q[..., 3] = np.where(positive, 0.25 * s, 0.0)
    q[..., 0] = np.where(positive, (m[..., 2, 1] - m[..., 1, 2]) / s, 0.0)
    q[..., 1] = np.where(positive, (m[..., 0, 2] - m[..., 2, 0]) / s, 0.0)
    q[..., 2] = np.where(positive, (m[..., 1, 0] - m[..., 0, 1]) / s, 0.0)
    for condition, (i, j, k) in (
        (~positive & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2]), (0, 1, 2)),
        (~positive & ~(m[..., 0, 0] > m[..., 1, 1]) & (m[..., 1, 1] > m[..., 2, 2]), (1, 2, 0)),
        (~positive & ~(m[..., 0, 0] > m[..., 1, 1]) & ~(m[..., 1, 1] > m[..., 2, 2]), (2, 0, 1)),
    ):
        s2 = np.sqrt(np.where(condition, 1.0 + m[..., i, i] - m[..., j, j] - m[..., k, k], 1.0)) * 2.0
        q[..., i] = np.where(condition, 0.25 * s2, q[..., i])
        q[..., j] = np.where(condition, (m[..., j, i] + m[..., i, j]) / s2, q[..., j])
        q[..., k] = np.where(condition, (m[..., k, i] + m[..., i, k]) / s2, q[..., k])
        q[..., 3] = np.where(condition, (m[..., k, j] - m[..., j, k]) / s2, q[..., 3])
    return q


def _world_deltas(quats: np.ndarray) -> dict:
    """SMPL-X 局部四元数 → 每个关节的累积世界增量矩阵 {(55, 3, 3)}。"""
    cache: dict[int, np.ndarray] = {}
    for index in range(len(SMPLX_PARENTS)):
        matrix = _mat_from_quat(quats[:, index, :])
        parent = SMPLX_PARENTS[index]
        cache[index] = matrix if parent < 0 else np.einsum("nij,njk->nik", cache[parent], matrix)
    return cache


def _ensure_continuity(values: np.ndarray) -> np.ndarray:
    """相邻帧四元数保持同半球，避免插值时翻转/抖动。"""
    out = np.array(values, dtype=np.float64, copy=True)
    for frame in range(1, out.shape[0]):
        dot = np.sum(out[frame] * out[frame - 1], axis=-1, keepdims=True)
        out[frame] = np.where(dot < 0.0, -out[frame], out[frame])
    return out



def _rodrigues(u, v):
    """把单位向量 u 旋到 v 的最小旋转（3x3）。"""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = v / max(np.linalg.norm(v), 1e-12)
    cos = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if cos > 1.0 - 1e-9:
        return np.eye(3)
    if cos < -1.0 + 1e-9:
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = np.cross(u, v)
    sin = np.linalg.norm(axis)
    axis = axis / sin
    angle = float(np.arctan2(sin, cos))
    x, y, z = axis
    c, s_ = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s_, x * z * C + y * s_],
        [y * x * C + z * s_, c + y * y * C, y * z * C - x * s_],
        [z * x * C - y * s_, z * y * C + x * s_, c + z * z * C],
    ])


def _rest_world(doc):
    """静止姿势下每个节点的世界旋转与位置（glTF 层级）。"""
    nodes = doc.get("nodes", [])
    parent = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []) or []:
            parent[int(child)] = index
    cache = {}

    def resolve(index):
        if index in cache:
            return cache[index]
        node = nodes[index]
        rotation = _mat_from_quat(np.asarray(node.get("rotation") or [0.0, 0.0, 0.0, 1.0], dtype=np.float64))
        translation = np.asarray(node.get("translation") or [0.0, 0.0, 0.0], dtype=np.float64)
        scale = np.asarray(node.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64)
        if index in parent:
            pr, pt, ps = resolve(parent[index])
            cache[index] = (pr @ rotation, pt + pr @ (ps * translation), ps * scale)
        else:
            cache[index] = (rotation, translation, scale)
        return cache[index]

    for index in range(len(nodes)):
        resolve(index)
    return cache


def _smplx_rest_joints():
    """SMPL-X 模板姿势的 55 个关节世界坐标（T-pose，米）；取不到返回 None。"""
    try:
        from .loader import models_root

        path = models_root() / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=True)
        template = np.asarray(data["v_template"], dtype=np.float64)
        regressor = data["J_regressor"]
        if hasattr(regressor, "dot"):
            joints = np.asarray(regressor.dot(template))
        else:
            joints = np.asarray(regressor, dtype=np.float64).dot(template)
        return joints[: len(SMPLX_PARENTS)]
    except Exception as exc:
        print(f"[zmocap] 读取 SMPL-X 模板失败：{exc}")
        return None



def _add_animation(appender: _Appender, doc: dict, mapping: dict, quats: np.ndarray,
                   fps: float, root_track=None) -> bool:
    """给映射到的骨骼节点加 rotation 动画轨道。

    做法是相似变换（similarity transform），不猜方向、不拟合：

    SMPL-X 的局部旋转 ``R_i`` 是相对 T-pose 的增量，沿骨链累乘得到世界增量
    ``D_i = D_parent · R_i``。要让角色骨骼的**世界朝向**等于「它的 rest 朝向再叠加
    同一个世界增量」，即 ``W_i(t) = W_rest(i) · D_i``，那么它的局部旋转必须是::

        L_i(t) = D_parent⁻¹ · R_rest(i) · D_i
               = D_parent⁻¹ · R_rest(i) · D_parent · R_i

    中间那一对 ``D_parent⁻¹ · D_parent`` 不是恒等——正是它把「角色的 rest 朝向」
    从「SMPL-X 的 rest 朝向」换算过去。漏掉它，误差会沿骨链一级级累积。
    """
    frames = int(quats.shape[0])
    times = (np.arange(frames, dtype=np.float32) / float(fps))
    time_accessor = appender.add_floats(times, "SCALAR", minmax=True)
    deltas = _world_deltas(quats)
    rest_world = _rest_world(doc)
    identity = np.eye(3, dtype=np.float64)

    samplers, channels = [], []
    for smplx_index, node_index in sorted(mapping.items()):
        local_source = _mat_from_quat(quats[:, smplx_index, :])
        rest = doc["nodes"][node_index].get("rotation") or [0.0, 0.0, 0.0, 1.0]
        rest_local = _mat_from_quat(np.asarray(rest, dtype=np.float64))

        smplx_parent = SMPLX_PARENTS[smplx_index] if smplx_index < len(SMPLX_PARENTS) else -1
        if smplx_parent >= 0 and smplx_parent in deltas:
            parent_delta = deltas[smplx_parent]
        else:
            parent_delta = np.broadcast_to(identity, local_source.shape)

        # L = D_parent⁻¹ · (R_rest · D_parent · R_i)
        rotated_rest = np.einsum("jk,nkl->njl", rest_local, parent_delta)
        local = np.einsum("nji,njk->nik", parent_delta, rotated_rest)
        local = np.einsum("nij,njk->nik", local, local_source)

        quats_out = _ensure_continuity(_quat_from_mat(local)).astype(np.float32)
        value_accessor = appender.add_floats(quats_out, "VEC4")
        samplers.append({"input": time_accessor, "output": value_accessor,
                         "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": int(node_index), "path": "rotation"}})
    if not channels:
        return False

    # 根骨位移：世界位移先按角色/SMPL-X 身高比缩放，再换算到父节点局部空间。
    pelvis = mapping.get(0)
    if pelvis is not None and root_track is not None:
        track = np.asarray(root_track, dtype=np.float64)
        rest_t = np.asarray(doc["nodes"][pelvis].get("translation") or [0.0, 0.0, 0.0],
                            dtype=np.float64)
        scale = _skeleton_scale(doc, mapping)
        offset = (track - track[0]) * scale
        parent_rotation, parent_scale = _parent_frame(doc, pelvis)
        local_offset = offset @ parent_rotation if parent_rotation is not None else offset
        if parent_scale is not None:
            local_offset = local_offset / np.where(np.abs(parent_scale) < 1e-9, 1.0, parent_scale)
        values = (rest_t.reshape(1, 3) + local_offset).astype(np.float32)
        value_accessor = appender.add_floats(values, "VEC3")
        samplers.append({"input": time_accessor, "output": value_accessor,
                         "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": int(pelvis), "path": "translation"}})

    doc.setdefault("animations", []).append(
        {"name": "zmocap", "samplers": samplers, "channels": channels}
    )
    return True


def _skeleton_scale(doc: dict, mapping: dict) -> float:
    """角色身高 / SMPL-X 身高；取不到模板姿势时退化为 1.0。"""
    rest_joints = _smplx_rest_joints()
    if rest_joints is None:
        return 1.0
    world = _rest_world(doc)
    ratios = []
    joints = np.asarray(rest_joints, dtype=np.float64)
    for child, parent in enumerate(SMPLX_PARENTS[:22]):
        if parent in mapping and child in mapping:
            source = np.linalg.norm(joints[child] - joints[parent])
            target = np.linalg.norm(world[mapping[child]][1] - world[mapping[parent]][1])
            if source > 1e-6 and target > 1e-6:
                ratios.append(target / source)
    return float(np.median(ratios)) if ratios else 1.0


def _depth(parent_map: dict, node_index: int) -> int:
    depth, current = 0, parent_map.get(node_index)
    while current is not None and depth < 64:
        depth += 1
        current = parent_map.get(current)
    return depth


def _parent_of(doc: dict, node_index: int):
    for index, node in enumerate(doc.get("nodes", [])):
        for child in node.get("children", []) or []:
            if int(child) == int(node_index):
                return index
    return None


def _parent_frame(doc: dict, node_index: int):
    """返回父节点累积的 rest 世界旋转与缩放；根节点返回 (None, None)。"""
    nodes = doc.get("nodes", [])
    parent: dict[int, int] = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []) or []:
            parent[int(child)] = index
    chain, current = [], parent.get(node_index)
    while current is not None:
        chain.append(current)
        current = parent.get(current)
    rotation = np.eye(3, dtype=np.float64)
    scale = np.ones(3, dtype=np.float64)
    for index in reversed(chain):
        node = nodes[index]
        rotation = rotation @ _mat_from_quat(np.asarray(node.get("rotation") or [0.0, 0.0, 0.0, 1.0],
                                                       dtype=np.float64))
        scale = scale * np.asarray(node.get("scale") or [1.0, 1.0, 1.0], dtype=np.float64)
    return rotation.T, scale
    return True


# SMPL-X 55 关节的父关节索引（-1 为根骨）
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
                 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20,
                 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]


def _add_bone_gizmos(appender: _Appender, joints: np.ndarray, fps: float,
                     radius: float = 0.022) -> int:
    """往现有 doc 里加一套“骨骼图层”（关节球 + 骨架线），带自带的平移动画。

    节点都带 ``extras.zmocapBone = true``，前端据此在蒙皮/骨骼显示层之间切换——
    这样切换纯属前端行为，不需要重跑任何节点。
    """
    import trimesh

    doc = appender.doc
    doc.setdefault("nodes", [])
    doc.setdefault("meshes", [])
    doc.setdefault("scenes", [{"nodes": []}])

    sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    positions = np.asarray(sphere.vertices, dtype=np.float32)
    indices = np.asarray(sphere.faces, dtype=np.uint32).reshape(-1)
    pos_accessor = appender.add_floats(positions, "VEC3", minmax=True)
    index_view = appender.add_view(indices.tobytes())
    doc["accessors"].append({"bufferView": index_view, "componentType": 5125,
                             "count": int(indices.size), "type": "SCALAR"})
    index_accessor = len(doc["accessors"]) - 1
    doc["meshes"].append({"primitives": [{"attributes": {"POSITION": pos_accessor},
                                          "indices": index_accessor, "mode": 4}]})
    mesh_index = len(doc["meshes"]) - 1

    count = joints.shape[0]
    rest = np.asarray(joints[0], dtype=np.float64)
    bone_ids: list[int] = []
    for index in range(55):
        parent = SMPLX_PARENTS[index]
        offset = rest[index] - (rest[parent] if parent >= 0 else 0.0)
        doc["nodes"].append({
            "name": f"zmocap_bone_{index}",
            "translation": [float(v) for v in offset],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "extras": {"zmocapBone": True},
        })
        bone_ids.append(len(doc["nodes"]) - 1)

    for index in range(55):
        doc["nodes"].append({"name": f"zmocap_bone_gizmo_{index}", "mesh": mesh_index,
                             "extras": {"zmocapBone": True}})
        doc["nodes"][bone_ids[index]].setdefault("children", []).append(len(doc["nodes"]) - 1)

    for index in range(55):
        parent = SMPLX_PARENTS[index]
        if parent >= 0:
            doc["nodes"][bone_ids[parent]].setdefault("children", []).append(bone_ids[index])

    scene = doc["scenes"][0]
    scene.setdefault("nodes", []).append(bone_ids[0])

    times = (np.arange(count, dtype=np.float32) / float(fps))
    time_accessor = appender.add_floats(times, "SCALAR", minmax=True)
    samplers, channels = [], []
    for index in range(55):
        parent = SMPLX_PARENTS[index]
        values = joints[:, index] - joints[:, parent] if parent >= 0 else joints[:, index]
        accessor = appender.add_floats(np.asarray(values, dtype=np.float32), "VEC3")
        samplers.append({"input": time_accessor, "output": accessor, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": bone_ids[index], "path": "translation"}})
    doc.setdefault("animations", []).append(
        {"name": "zmocap_bones", "samplers": samplers, "channels": channels}
    )
    return bone_ids[0]


def _build_mannequin(appender: _Appender, joints: np.ndarray, quats: np.ndarray, fps: float) -> dict:
    """兜底骨架：真正的骨骼层级 + 每关节一个球。

    动画**只写父子相对位移**（不写旋转）：第 0 帧骨架是 rest 姿势，后续每帧把
    ``joints[t, i] - joints[t, parent]`` 写成该节点的局部平移。这样前向运动学的结果
    与 SMPL-X 关节逐帧严格一致——之前用「第 0 帧当 rest + 绝对旋转」的做法会让
    手指这类远离根节点的地方偏到脸的侧面去。
    """
    import trimesh

    doc = appender.doc
    doc.setdefault("nodes", [])
    doc.setdefault("meshes", [])
    doc.setdefault("scenes", [{"nodes": []}])

    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    positions = np.asarray(sphere.vertices, dtype=np.float32)
    indices = np.asarray(sphere.faces, dtype=np.uint32).reshape(-1)
    pos_accessor = appender.add_floats(positions, "VEC3", minmax=True)
    index_view = appender.add_view(indices.tobytes())
    doc["accessors"].append({"bufferView": index_view, "componentType": 5125,
                             "count": int(indices.size), "type": "SCALAR"})
    index_accessor = len(doc["accessors"]) - 1
    doc["meshes"].append({"primitives": [{"attributes": {"POSITION": pos_accessor},
                                          "indices": index_accessor, "mode": 4}]})

    count = joints.shape[0]
    rest = np.asarray(joints[0], dtype=np.float64)
    bone_ids = []
    for index in range(55):
        parent = SMPLX_PARENTS[index]
        offset = rest[index] - (rest[parent] if parent >= 0 else 0.0)
        doc["nodes"].append({
            "name": SMPLX_NAMES[index] if index < len(SMPLX_NAMES) else f"J{index}",
            "translation": [float(v) for v in offset],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        })
        bone_ids.append(len(doc["nodes"]) - 1)

    for index in range(55):
        doc["nodes"].append({"name": f"M{index}", "mesh": 0})
        sphere_id = len(doc["nodes"]) - 1
        doc["nodes"][bone_ids[index]].setdefault("children", []).append(sphere_id)

    for index in range(55):
        parent = SMPLX_PARENTS[index]
        if parent >= 0:
            doc["nodes"][bone_ids[parent]].setdefault("children", []).append(bone_ids[index])
    doc["scenes"][0]["nodes"] = [bone_ids[0]]

    times = (np.arange(count, dtype=np.float32) / float(fps))
    time_accessor = appender.add_floats(times, "SCALAR", minmax=True)

    samplers, channels = [], []
    for index in range(55):
        parent = SMPLX_PARENTS[index]
        if parent >= 0:
            values = joints[:, index] - joints[:, parent]
        else:
            values = joints[:, index]
        accessor = appender.add_floats(np.asarray(values, dtype=np.float32), "VEC3")
        samplers.append({"input": time_accessor, "output": accessor, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": bone_ids[index], "path": "translation"}})

    doc.setdefault("animations", []).append(
        {"name": "zmocap", "samplers": samplers, "channels": channels}
    )
    return doc


def build_glb(motion: dict, out_path: Path, character: str = "", mode: str = "auto",
              strict: bool = False, bones: bool = True) -> tuple[Path, dict]:
    """把动作写成 GLB；返回 (路径, 统计信息)。"""
    import numpy as np

    joints = np.asarray(motion["joints"], dtype=np.float32)[:, :55]
    fps = float(motion.get("fps", 30.0))
    quats = local_rotations(motion["params"])

    character = str(character or "").strip()
    if character and mode != "mannequin":
        from .character import resolve_character

        character = resolve_character(character)
    info = {"mode": "mannequin", "bones": 0, "frames": int(joints.shape[0])}

    if character and mode != "mannequin":
        try:
            doc, binary = _read_glb(Path(character).expanduser())
            appender = _Appender(doc, binary)
            mapping = _bone_map(doc)
            if mapping and _add_animation(appender, doc, mapping, quats, fps, root_track=joints[:, 0]):
                if bones:
                    # 骨酪图层与角色共存于同一个 GLB：前端切显示层不用重跑工作流。
                    _add_bone_gizmos(appender, joints, fps)
                appender.finish()
                info.update({"mode": "character", "bones": len(mapping)})
                return _write_glb(Path(out_path), doc, appender.binary), info
            if strict:
                raise RuntimeError(
                    f"角色 {Path(character).name} 没有匹配到人形骨骼（支持 Mixamo / VRM 命名）；"
                    "把 skinned 关掉可用自带骨架")
            print("[zmocap] 角色骨架名未匹配（支持 Mixamo / VRM 命名），回退到自带骨架")
        except Exception as exc:
            print(f"[zmocap] 载入角色失败（{exc}），回退到自带骨架")

    doc = {"asset": {"version": "2.0", "generator": "zmocap"}}
    appender = _Appender(doc, bytearray())
    _build_mannequin(appender, joints, quats, fps)
    appender.finish()
    info.update({"mode": "mannequin", "bones": len(SMPLX_PARENTS)})
    return _write_glb(Path(out_path), doc, appender.binary), info


class ZMocapRig:
    """把动作绑到角色（或自带骨架）上，输出带骨骼图层的 GLB，并在节点内直接预览。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_MOTION",),
                "filename": ("STRING", {"default": "zmocap_rig"}),
            },
            "optional": {
                # 角色完全由 `zmocap VRM Loader` 提供；不接就是自带骨架。
                "character": ("ZMOCAP_CHARACTER",),
                # 仅影响前端显示层：True = 蒙皮网格，False = 骨骼。切换即时生效，
                # 不触发重跑（两套几何都在同一个 GLB 里）。
                "skinned": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, motion, filename, character=None, skinned=True):
        import hashlib

        import folder_paths

        if isinstance(character, dict):
            target = str(character.get("path") or "")
        else:
            target = str(character or "").strip()
        if target.startswith("("):
            target = ""
        # 规范 humanoid 运动（skeleton='vrm'）：绑定/写文件都交给 binding 模块，
        # 角色只在这一步用得上，转换类节点完全不碰角色。
        if str(motion.get("skeleton") or "") == "vrm":
            from ..soma.binding import write_character_glb, write_humanoid_glb

            animation = np.asarray(motion["world_rotation"], dtype=np.float64)
            seed = "|".join([
                "vrm", str(animation.shape[0]), f"{float(motion.get('fps', 30.0)):.3f}",
                f"{float(np.abs(animation).sum()):.6f}", f"{float(np.abs(motion['transl']).sum()):.6f}",
                target,
            ])
            digest = hashlib.sha1(seed.encode()).hexdigest()[:8]
            outdir = Path(folder_paths.get_output_directory())
            out_path = outdir / f"{filename}_{digest}.glb"
            if target:
                path, report = write_character_glb(motion, target, out_path)
            else:
                path, report = write_humanoid_glb(motion, out_path)
            print(f"[zmocap] rig {Path(path).name} ({report['bones']} bones, "
                  f"{report.get('frames', int(animation.shape[0]))} frames, scale {report.get('scale', 1.0)})")
            relative = path.relative_to(outdir).as_posix()
            return {
                "ui": {"zmocap_glb": [relative], "zmocap_rig": [{"path": str(path), **report}]},
                "result": (relative,),
            }

        # 有角色就一律按角色绑（`skinned` 只是前端的显示层开关，不再影响绑定内容），
        # 否则输出自带骨架。
        mode = "auto" if target else "mannequin"
        # 输出名带内容指纹：改了角色/模式/动作就换 URL，前端 three.js 必然重新加载，
        # 否则同一个文件名会被浏览器缓存，切换看起来“没生效”。
        params = motion.get("params") or {}
        seed = "|".join([
            str(motion.get("source")), str(int(np.asarray(motion["joints"]).shape[0])),
            f"{float(motion.get('fps', 30.0)):.3f}", target,
            f"{float(np.abs(np.asarray(motion['joints'])).sum()):.4f}",
            str(int(np.asarray(params.get('betas')).sum() * 1000)) if 'betas' in params else "",
        ])
        digest = hashlib.sha1(seed.encode()).hexdigest()[:8]
        outdir = Path(folder_paths.get_output_directory())
        path, info = build_glb(motion, outdir / f"{filename}_{digest}.glb",
                               character=target, mode=mode, strict=bool(target), bones=True)
        print(f"[zmocap] rig {info['mode']} -> {path} ({info['bones']} bones, {info['frames']} frames)")
        # Preview3D 只认 output 目录下的相对文件名（前端用 /view?filename=... 加载），
        # 绝对路径会 404 变成空白预览，所以返回值给相对路径，绝对路径放在 ui 里。
        relative = path.relative_to(Path(folder_paths.get_output_directory())).as_posix()
        return {
            # zmocap_glb 给节点内置的 three.js 预览用（相对 output 的路径 + 指纹，避免缓存）
            "ui": {"zmocap_glb": [relative], "zmocap_rig": [{"path": str(path), **info}]},
            "result": (relative,),
        }
