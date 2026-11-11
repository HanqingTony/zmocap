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

# 目标骨架候选名：Mixamo（FBX/GLB 常见）与 VRM humanoid
_ALIASES = {
    "pelvis": ["hips", "mixamorighips", "pelvis", "j_bip_c_hips"],
    "spine1": ["spine", "mixamorigspine", "spine1", "j_bip_c_spine"],
    "spine2": ["spine1", "mixamorigspine1", "spine2", "j_bip_c_chest"],
    "spine3": ["spine2", "mixamorigspine2", "spine3", "j_bip_c_upperchest"],
    "neck": ["neck", "mixamorigneck", "j_bip_c_neck"],
    "head": ["head", "mixamorighead", "j_bip_c_head"],
    "left_shoulder": ["leftshoulder", "mixamorigleftshoulder", "left_shoulder", "j_bip_l_shoulder"],
    "right_shoulder": ["rightshoulder", "mixamorigrightshoulder", "right_shoulder", "j_bip_r_shoulder"],
    "left_elbow": ["leftarm", "mixamorigleftarm", "left_elbow", "j_bip_l_lowerarm"],
    "right_elbow": ["rightarm", "mixamorigrightarm", "right_elbow", "j_bip_r_lowerarm"],
    "left_wrist": ["lefthand", "mixamoriglefthand", "left_wrist", "j_bip_l_hand"],
    "right_wrist": ["righthand", "mixamorigrighthand", "right_wrist", "j_bip_r_hand"],
    "left_hip": ["leftupleg", "mixamorigleftupleg", "left_hip", "j_bip_l_upperleg"],
    "right_hip": ["rightupleg", "mixamorigrightupleg", "right_hip", "j_bip_r_upperleg"],
    "left_knee": ["leftleg", "mixamorigleftleg", "left_knee", "j_bip_l_lowerleg"],
    "right_knee": ["rightleg", "mixamorigrightleg", "right_knee", "j_bip_r_lowerleg"],
    "left_ankle": ["leftfoot", "mixamorigleftfoot", "left_ankle", "j_bip_l_foot"],
    "right_ankle": ["rightfoot", "mixamorigrightfoot", "right_ankle", "j_bip_r_foot"],
}

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
    """SMPL-X 关节索引 → 目标 GLB 的节点索引。"""
    lookup = {}
    for index, node in enumerate(doc.get("nodes", [])):
        name = node.get("name")
        if name:
            lookup.setdefault(_normalize(name), index)
    mapping = {}
    for smplx_index, smplx_name in enumerate(SMPLX_NAMES):
        for alias in _ALIASES.get(smplx_name, []):
            node_index = lookup.get(_normalize(alias))
            if node_index is not None:
                mapping[smplx_index] = node_index
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


def _add_animation(appender: _Appender, doc: dict, mapping: dict, quats: np.ndarray, fps: float) -> bool:
    """给映射到的骨骼节点加 rotation 动画轨道。

    SMPL-X 的局部旋转是“相对静止姿势的增量”，而角色骨骼的静止旋转通常不是单位
    四元数，所以要先取出骨骼的 rest 旋转再右乘增量：q = q_rest ⊗ q_delta。
    """
    times = (np.arange(quats.shape[0], dtype=np.float32) / float(fps))
    time_accessor = appender.add_floats(times, "SCALAR", minmax=True)
    samplers, channels = [], []
    for smplx_index, node_index in sorted(mapping.items()):
        delta = np.asarray(quats[:, smplx_index, :], dtype=np.float64)
        rest = doc["nodes"][node_index].get("rotation") or [0.0, 0.0, 0.0, 1.0]
        rest = np.asarray(rest, dtype=np.float64).reshape(1, 4)
        values = _quat_mul(np.repeat(rest, delta.shape[0], axis=0), delta).astype(np.float32)
        # glTF 四元数顺序是 (x, y, z, w)，与我们的输出一致
        value_accessor = appender.add_floats(values, "VEC4")
        samplers.append({"input": time_accessor, "output": value_accessor,
                         "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": int(node_index), "path": "rotation"}})
    if not channels:
        return False
    doc.setdefault("animations", []).append(
        {"name": "zmocap", "samplers": samplers, "channels": channels}
    )
    return True


def _build_mannequin(appender: _Appender, joints: np.ndarray, fps: float) -> dict:
    """兜底骨架：55 个球 + 每帧平移动画（不依赖任何外部模型）。"""
    doc = appender.doc
    doc.setdefault("nodes", [])
    doc.setdefault("meshes", [])
    doc.setdefault("scenes", [{"nodes": []}])

    # 一个低模球体反复复用
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
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
    times = (np.arange(count, dtype=np.float32) / float(fps))
    time_accessor = appender.add_floats(times, "SCALAR", minmax=True)

    node_ids = []
    for index in range(55):
        doc["nodes"].append({"name": f"J{index}", "mesh": 0})
        node_ids.append(len(doc["nodes"]) - 1)
    doc["scenes"][0]["nodes"] = [node_ids[0]]

    samplers, channels = [], []
    for index in range(55):
        track = np.asarray(joints[:, index, :], dtype=np.float32)
        accessor = appender.add_floats(track, "VEC3")
        samplers.append({"input": time_accessor, "output": accessor, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": node_ids[index], "path": "translation"}})
    doc.setdefault("animations", []).append(
        {"name": "zmocap", "samplers": samplers, "channels": channels}
    )
    return doc


def build_glb(motion: dict, out_path: Path, character: str = "", mode: str = "auto") -> tuple[Path, dict]:
    """把动作写成 GLB；返回 (路径, 统计信息)。"""
    import numpy as np

    joints = np.asarray(motion["joints"], dtype=np.float32)[:, :55]
    fps = float(motion.get("fps", 30.0))
    quats = local_rotations(motion["params"])

    character = str(character or "").strip()
    info = {"mode": "mannequin", "bones": 0, "frames": int(joints.shape[0])}

    if character and mode != "mannequin":
        try:
            doc, binary = _read_glb(Path(character).expanduser())
            appender = _Appender(doc, binary)
            mapping = _bone_map(doc)
            if mapping and _add_animation(appender, doc, mapping, quats, fps):
                appender.finish()
                info.update({"mode": "character", "bones": len(mapping)})
                return _write_glb(Path(out_path), doc, appender.binary), info
            print("[zmocap] 角色骨架名未匹配（支持 Mixamo / VRM 命名），回退到自带骨架")
        except Exception as exc:
            print(f"[zmocap] 载入角色失败（{exc}），回退到自带骨架")

    doc = {"asset": {"version": "2.0", "generator": "zmocap"}}
    appender = _Appender(doc, bytearray())
    _build_mannequin(appender, joints, fps)
    appender.finish()
    return _write_glb(Path(out_path), doc, appender.binary), info


class ZMocapRig:
    """把动作绑到角色（或自带骨架）上，输出可被官方 `Preview3D` 播放的 GLB。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion": ("ZMOCAP_MOTION",),
                "filename": ("STRING", {"default": "zmocap_rig"}),
            },
            "optional": {
                "character": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "角色 GLB/VRM 路径；留空用自带骨架"}),
                "mode": (["auto", "mannequin"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, motion, filename, character="", mode="auto"):
        import folder_paths

        outdir = Path(folder_paths.get_output_directory())
        path, info = build_glb(motion, outdir / f"{filename}.glb", character=character, mode=mode)
        print(f"[zmocap] rig {info['mode']} → {path} ({info['bones']} bones, {info['frames']} frames)")
        return {"ui": {"zmocap_rig": [{"path": str(path), **info}]}, "result": (str(path),)}
