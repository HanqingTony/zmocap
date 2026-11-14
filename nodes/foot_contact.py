"""触地/触物检测：按**通道**输出接触时间轴（``zmocap.contact.v2``）。

通道默认是左右脚（用 ExMR 的 ``static_conf_logits`` 做置信度），
其余部位（手、膝、骨盆、头）用「最低顶点高度 + 速度」几个几何门限判断。
每个通道都能在节点上单独开关。

同一套阈值：``static_conf``（置信度/几何分数）、``height_tolerance``（离地高度）、
``speed_limit``（水平速度）、``vertical_limit``（垂直速度）、``min_frames``（最短区间）、
``gap_frames``（短间隙合并）。
"""

from __future__ import annotations

import hashlib

import numpy as np

CONTACT_FORMAT = "zmocap.contact.v2"
CONTACT_VERSION = "2.0.0"

# 通道定义：关节用于取“接触点位置”，vertices 用于取“最低顶点高度”。
# SMPL-X 关节编号：0 骨盆、1/2 髋、4/5 膝、7/8 踝、10/11 脚、20/21 腕、25-54 手指、12 颈、15 头
CHANNELS: dict[str, dict] = {
    "foot_left": {"label": "左脚", "joints": (7, 10), "reps": (7, 10), "color": "#3b82f6",
                  "vertices": (7, 10), "logits": (0, 1), "primary": 10},
    "foot_right": {"label": "右脚", "joints": (8, 11), "reps": (8, 11), "color": "#ef4444",
                   "vertices": (8, 11), "logits": (2, 3), "primary": 11},
    "hand_left": {"label": "左手", "joints": (20,) + tuple(range(25, 40)), "reps": (20, 25), "color": "#22c55e",
                  "vertices": (20,) + tuple(range(25, 40)), "logits": None, "primary": 20},
    "hand_right": {"label": "右手", "joints": (21,) + tuple(range(40, 55)), "reps": (21, 40), "color": "#f59e0b",
                   "vertices": (21,) + tuple(range(40, 55)), "logits": None, "primary": 21},
    "knee_left": {"label": "左膝", "joints": (1, 4), "reps": (4, 7), "color": "#a855f7",
                  "vertices": (1, 4), "logits": None, "primary": 4},
    "knee_right": {"label": "右膝", "joints": (2, 5), "reps": (5, 8), "color": "#ec4899",
                   "vertices": (2, 5), "logits": None, "primary": 5},
    "pelvis": {"label": "骨盆", "joints": (0,), "reps": (0, 1), "color": "#eab308",
               "vertices": (0, 1, 2), "logits": None, "primary": 0},
    "head": {"label": "头部", "joints": (12, 15, 22, 23, 24), "reps": (12, 15), "color": "#e5e7eb",
             "vertices": (12, 15, 22, 23, 24), "logits": None, "primary": 15},
}
CHANNEL_ORDER = tuple(CHANNELS)

_BODY = None
_VERTEX_MASKS: dict | None = None


def _body_model():
    global _BODY
    if _BODY is not None:
        return _BODY
    import sys

    from .loader import REPO, models_root

    vendor = REPO / "vendor/ExMR"
    if not (vendor / "exmr").is_dir():
        raise RuntimeError(f"ExMR 代码缺失: {vendor}（vendor/ExMR 应随仓库部署）")
    for path in (vendor, REPO / "support"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from exmr.body_models.smplx_model import SmplxModel

    body = models_root() / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not body.is_file():
        raise FileNotFoundError(f"缺 SMPL-X 模型：{body}")
    _BODY = SmplxModel(str(body))
    _BODY.eval()
    return _BODY


def _vertex_masks() -> dict:
    """每个通道的顶点索引：取“主导关节”落在该通道关节集合里的顶点。"""
    global _VERTEX_MASKS
    if _VERTEX_MASKS is not None:
        return _VERTEX_MASKS
    from .loader import models_root

    path = models_root() / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    data = np.load(str(path), allow_pickle=True)
    weights = np.asarray(data["weights"], dtype=np.float32)  # (V, 55)
    dominant = weights.argmax(axis=1)
    masks = {}
    for name, spec in CHANNELS.items():
        index = np.flatnonzero(np.isin(dominant, np.asarray(spec["vertices"], dtype=np.int64)))
        masks[name] = index.astype(np.int64)
    _VERTEX_MASKS = masks
    return masks


def _batch(value):
    import torch

    tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(np.asarray(value))
    return tensor.to(torch.float32).reshape(tensor.shape[0], -1)[None]


def _logits(value, frames):
    import torch

    tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(np.asarray(value))
    return tensor.to(torch.float32).reshape(int(frames), -1)


def _intervals(mask):
    edges = np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8))
    return [[int(s), int(e)] for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _fill_gaps(mask, gap):
    if gap <= 0:
        return mask
    out = mask.copy()
    length, index = len(out), 0
    while index < length:
        if out[index]:
            index += 1
            continue
        end = index
        while end < length and not out[end]:
            end += 1
        if 0 < index and end < length and end - index <= gap:
            out[index:end] = True
        index = end
    return out


def _drop_short(mask, minimum):
    if minimum <= 1:
        return mask
    out = mask.copy()
    for start, end in _intervals(out):
        if end - start < minimum:
            out[start:end] = False
    return out


def _speed(points, fps):
    """(T, C, 2) XZ → (T, C) m/s。"""
    if len(points) <= 1:
        return np.zeros(points.shape[:2], dtype=np.float32)
    delta = np.linalg.norm(np.diff(points, axis=0), axis=-1) * float(fps)
    return np.vstack([delta[:1], delta]).astype(np.float32)


def _vertical_speed(heights, fps):
    if len(heights) <= 1:
        return np.zeros_like(heights, dtype=np.float32)
    delta = np.abs(np.diff(heights, axis=0)) * float(fps)
    return np.vstack([delta[:1], delta]).astype(np.float32)


def _motion_hash(motion):
    digest = hashlib.sha256()
    for key in ("betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose"):
        value = motion["params"].get(key)
        if value is None:
            continue
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    digest.update(f"{float(motion.get('fps', 30.0)):.6f}".encode())
    return digest.hexdigest()


def _normalize_ranges(ranges, frames, channels):
    result = []
    for index in range(channels):
        track = (ranges or [])[index] if index < len(ranges or []) else []
        normalized = []
        for pair in sorted(track):
            if len(pair) != 2:
                continue
            start = max(0, min(frames, int(pair[0])))
            end = max(0, min(frames, int(pair[1])))
            if start >= end:
                continue
            if normalized and start <= normalized[-1][1]:
                normalized[-1] = [normalized[-1][0], max(normalized[-1][1], end)]
            else:
                normalized.append([start, end])
        result.append(normalized)
    return result


def _mask_from_ranges(ranges, frames, channels):
    mask = np.zeros((frames, channels), dtype=np.uint8)
    for channel, track in enumerate(ranges):
        for start, end in track:
            mask[start:end, channel] = 1
    return mask


class ZMocapFootContact:
    """按通道检测接触（脚/手/膝/骨盆/头），输出 ``zmocap.contact.v2``。"""

    @classmethod
    def INPUT_TYPES(cls):
        toggles = {
            "foot_left": ("BOOLEAN", {"default": True}),
            "foot_right": ("BOOLEAN", {"default": True}),
            "hand_left": ("BOOLEAN", {"default": False}),
            "hand_right": ("BOOLEAN", {"default": False}),
            "knee_left": ("BOOLEAN", {"default": False}),
            "knee_right": ("BOOLEAN", {"default": False}),
            "pelvis": ("BOOLEAN", {"default": False}),
            "head": ("BOOLEAN", {"default": False}),
        }
        return {
            "required": {"motion": ("ZMOCAP_SMPLX",), **toggles},
            "optional": {
                "static_conf": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "height_tolerance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005}),
                "speed_limit": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.05}),
                "vertical_limit": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05}),
                "min_frames": ("INT", {"default": 2, "min": 1, "max": 240, "step": 1}),
                "gap_frames": ("INT", {"default": 2, "min": 0, "max": 240, "step": 1}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_SMPLX", "ZMOCAP_CONTACTS")
    RETURN_NAMES = ("motion", "contacts")
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "按通道检测触地：可勾选脚/手/膝/骨盆/头，输出 zmocap.contact.v2。"

    def run(self, motion, foot_left=True, foot_right=True, hand_left=False, hand_right=False,
            knee_left=False, knee_right=False, pelvis=False, head=False,
            static_conf=0.8, height_tolerance=0.05, speed_limit=0.7, vertical_limit=0.8,
            min_frames=2, gap_frames=2):
        skeleton = str(motion.get("skeleton") or ("smplx" if "params" in motion else ""))
        if skeleton != "smplx":
            raise ValueError(f"Foot Contact 需要 SMPL-X motion，收到 {skeleton or '未知'}")

        enabled = [name for name, flag in zip(
            CHANNEL_ORDER,
            (foot_left, foot_right, hand_left, hand_right, knee_left, knee_right, pelvis, head),
        ) if flag]
        if not enabled:
            raise ValueError("至少勾选一个接触通道")

        import torch

        body = _body_model()
        from exmr.model.global_traj import lowest_vertex_height

        params = {
            key: _batch(motion["params"][key])
            for key in ("betas", "body_pose", "global_orient", "transl",
                        "left_hand_pose", "right_hand_pose")
            if key in motion["params"]
        }
        frames = int(params["transl"].shape[1])
        fps = float(motion.get("fps", 30.0))
        masks = _vertex_masks()

        groups = [masks[name] for name in enabled] + [slice(None)]
        with torch.inference_mode():
            heights = lowest_vertex_height(params, body, groups)[0].numpy().astype(np.float32)

        per_channel = heights[:, :len(enabled)]
        # 每帧全身最低点当作该帧的地面：脚/手/膝都要跟它比，才是“接触地面”。
        body_lowest = heights[:, len(enabled)]
        floor = float(body_lowest.min())
        height = per_channel - body_lowest[:, None]

        # 置信度：脚用 ExMR 的 static_conf_logits；其余部位纯几何判定（gates=1）。
        logits = motion.get("static_conf_logits")
        gates = np.ones_like(height)
        if logits is not None:
            conf = torch.sigmoid(_logits(logits, frames)).numpy()
            for index, name in enumerate(enabled):
                columns = [c for c in (CHANNELS[name]["logits"] or ()) if c < conf.shape[1]]
                if columns:
                    gates[:, index] = conf[:, columns].max(axis=1)

        joints = np.asarray(motion.get("joints"), dtype=np.float32) if motion.get("joints") is not None else None
        if joints is not None and joints.ndim == 3:
            frames = min(frames, joints.shape[0])
            height = height[:frames]
            gates = gates[:frames]
            points = np.zeros((frames, len(enabled), 2), dtype=np.float32)
            for index, name in enumerate(enabled):
                reps = [j for j in CHANNELS[name]["reps"] if j < joints.shape[1]]
                points[:, index] = joints[:frames][:, reps].mean(axis=1)[:, [0, 2]]
        else:
            points = np.zeros((frames, len(enabled), 2), dtype=np.float32)

        tolerance = max(float(height_tolerance), 1e-6)
        speed = _speed(points, fps)
        vertical = _vertical_speed(height, fps)
        height_score = np.clip(1.0 - height / tolerance, 0.0, 1.0)
        geometric = (height <= tolerance) & (speed < float(speed_limit)) & (vertical < float(vertical_limit))
        contact = geometric & (gates >= float(static_conf))
        probability = np.where(gates < 1.0 - 1e-6, gates * height_score, height_score).astype(np.float32)

        for index in range(len(enabled)):
            contact[:, index] = _drop_short(
                _fill_gaps(contact[:, index], int(gap_frames)), int(min_frames)
            )

        ranges = [_intervals(contact[:, index]) for index in range(len(enabled))]
        parts = {}
        for index, name in enumerate(enabled):
            parts[name] = {
                "label": CHANNELS[name]["label"],
                "color": CHANNELS[name]["color"],
                "ranges": ranges[index],
                "frames": int(contact[:, index].sum()),
            }

        contacts = {
            "format": CONTACT_FORMAT,
            "fps": fps,
            "frames": int(frames),
            "channels": list(enabled),
            "space": {"frame": "world", "up": "Y", "unit": "m"},
            "floor": {"mode": "plane", "height": floor},
            "parts": parts,
            "arrays": {
                "contact": contact.astype(np.uint8),
                "probability": probability,
                "height": height.astype(np.float32),
                "level": np.full(frames, floor, dtype=np.float32),
            },
            "meta": {
                "source": "auto",
                "algorithm": "lowest_vertex_height+velocity+static_conf",
                "thresholds": {
                    "static_conf": float(static_conf),
                    "height_tolerance": float(height_tolerance),
                    "speed_limit": float(speed_limit),
                    "vertical_limit": float(vertical_limit),
                    "min_frames": int(min_frames),
                    "gap_frames": int(gap_frames),
                },
                "channels": {
                    name: {"joints": list(CHANNELS[name]["joints"]), "color": CHANNELS[name]["color"],
                           "label": CHANNELS[name]["label"]}
                    for name in enabled
                },
                "version": CONTACT_VERSION,
                "motion_sha256": _motion_hash(motion),
            },
        }
        summary = ", ".join(f"{CHANNELS[n]['label']} {len(parts[n]['ranges'])} 段/{parts[n]['frames']} 帧" for n in enabled)
        print(f"[zmocap] contact: {frames} 帧 | {summary} | floor {floor:.3f} m")
        from ..motion import make_clip

        return (make_clip(motion, "smplx", sources=motion.get("sources") or {}), contacts)


class ZMocapContactLoad:
    """读取手改后的 ``zmocap.contact.v2`` JSON，覆盖自动区间。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"path": ("STRING", {"default": ""})},
            "optional": {"motion": ("ZMOCAP_SMPLX",)},
        }

    RETURN_TYPES = ("ZMOCAP_CONTACTS",)
    RETURN_NAMES = ("contacts",)
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "读回手改过的 contacts.json（zmoap.contact.v2），按其中的区间覆盖自动结果。"

    def run(self, path, motion=None):
        import json
        from pathlib import Path

        target = Path(str(path)).expanduser()
        if not target.is_file():
            raise FileNotFoundError(f"找不到 contacts 文件：{target}")
        data = json.loads(target.read_text())
        if data.get("format") != CONTACT_FORMAT:
            raise ValueError(f"不支持的触地格式：{data.get('format')!r}（需要 {CONTACT_FORMAT}）")
        channels = list(data.get("channels") or [])
        frames = int(data.get("frames") or 0)
        if not channels or frames <= 0:
            raise ValueError("contacts 文件缺少 channels/frames")
        ranges = [list((data.get("parts") or {}).get(name, {}).get("ranges") or []) for name in channels]
        ranges = _normalize_ranges(ranges, frames, len(channels))
        data["arrays"] = dict(data.get("arrays") or {})
        data["arrays"]["contact"] = _mask_from_ranges(ranges, frames, len(channels))
        for name, track in zip(channels, ranges):
            data.setdefault("parts", {}).setdefault(name, {})["ranges"] = track
        data.setdefault("meta", {})["source"] = "manual"
        print(f"[zmocap] contact 载入 {target.name}: {len(channels)} 通道 / {frames} 帧（manual）")
        return (data,)


def validate_contacts(contacts, motion) -> list[str]:
    """按 ``zmocap.contact.v2`` 校验；返回警告列表（不致命的问题）。"""
    if not isinstance(contacts, dict):
        raise ValueError("触地数据不是字典")
    fmt = contacts.get("format")
    if fmt != CONTACT_FORMAT:
        raise ValueError(f"触地数据格式不匹配：期望 {CONTACT_FORMAT}，实际 {fmt}")
    arrays = contacts.get("arrays") or {}
    contact = arrays.get("contact")
    if contact is None:
        raise ValueError("触地数据缺少 arrays.contact")
    contact = np.asarray(contact)
    channels = list(contacts.get("channels") or [])
    if contact.ndim != 2:
        raise ValueError(f"arrays.contact 形状应为 (T,C)，实际 {contact.shape}")
    if channels and contact.shape[1] != len(channels):
        raise ValueError(f"channels 数量 {len(channels)} 与 contact 通道 {contact.shape[1]} 不一致")

    warnings: list[str] = []
    frames = int(contacts.get("frames", contact.shape[0]))
    if frames != int(contact.shape[0]):
        warnings.append(f"frames={frames} 与 arrays.contact 长度 {contact.shape[0]} 不一致")
    motion_frames = len(np.asarray(motion.get("joints", [])))
    if motion_frames and frames != motion_frames:
        warnings.append(f"触地有 {frames} 帧，动作有 {motion_frames} 帧（将按较短者对齐）")
    fps = float(contacts.get("fps", 0.0))
    motion_fps = float(motion.get("fps", 0.0))
    if fps and motion_fps and abs(fps - motion_fps) > 0.01:
        warnings.append(f"触地 fps={fps:.2f} 与动作 fps={motion_fps:.2f} 不同")
    expected = (contacts.get("meta") or {}).get("motion_sha256")
    if expected and motion_frames:
        try:
            actual = _motion_hash(motion)
        except Exception:
            return warnings
        if expected != actual:
            warnings.append("触地数据的 motion_sha256 与当前动作不一致（可能来自别的视频）")
    return warnings


def ring_payload(contacts, motion) -> dict:
    """预览用：每个通道每帧的着地位置与开关。"""
    if "joints" not in motion:
        raise ValueError("触地光圈需要带关节序列的 SMPL-X 运动（VRM 运动只有旋转，无法定位地面点）")
    warnings = validate_contacts(contacts, motion)
    joints = np.asarray(motion["joints"], dtype=np.float64)
    channels = list(contacts.get("channels") or ["foot_left", "foot_right"])
    mask = np.asarray((contacts.get("arrays") or {})["contact"]) > 0
    frames = min(joints.shape[0], mask.shape[0])
    floor = float((contacts.get("floor") or {}).get("height", 0.0))

    out = {}
    for column, name in enumerate(channels):
        spec = CHANNELS.get(name, {})
        reps = [j for j in spec.get("reps", (10,)) if j < joints.shape[1]] or [10]
        points = joints[:frames][:, reps].mean(axis=1)
        position = np.stack([points[:, 0], np.full(frames, floor), points[:, 2]], axis=1)
        out[name] = {
            "label": spec.get("label", name),
            "color": spec.get("color", "#ffffff"),
            "on": mask[:frames, column].astype(int).tolist(),
            "pos": np.round(position, 4).tolist(),
        }

    payload = {
        "format": CONTACT_FORMAT,
        "fps": float(contacts.get("fps", motion.get("fps", 30.0))),
        "frames": int(frames),
        "floor": floor,
        "channels": out,
        "warnings": warnings,
    }
    # 兼容旧前端：左右脚走 left/right 键
    if "foot_left" in out:
        payload["left"] = {"on": out["foot_left"]["on"], "pos": out["foot_left"]["pos"]}
    if "foot_right" in out:
        payload["right"] = {"on": out["foot_right"]["on"], "pos": out["foot_right"]["pos"]}
    return payload
