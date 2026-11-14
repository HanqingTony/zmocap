"""脚接触检测：ExMR 的 static_conf_logits + SMPL-X 脚底高度 → 左右脚 contact 时间轴。

输出两样东西：

* ``motion``：原样透传（ExMR 的 static_joint_transl / static_joint_ik / put_on_ground
  已经在模型里做过修正，这就是“修正后的动作”）；
* ``contacts``：``zmocap.contact.v1`` 契约的数据字典，含区间、逐帧数组、地面、元数据。

不做时间轴 UI：数据交给 ``ZMocapExport`` 写成 JSON/NPZ，需要手动修时改 JSON，
再用 ``ZMocapContactLoad`` 读回来（source=manual）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CONTACT_FORMAT = "zmocap.contact.v1"
CONTACT_VERSION = "1.0.0"

# SMPL-X 身体模型只加载一次（CPU，约 3 秒；之后每次检测约 0.1 秒/百帧）。
_BODY = None


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


def _batch(value):
    """(T, D) torch/numpy → (1, T, D) float32 torch。"""
    import torch

    tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(np.asarray(value))
    tensor = tensor.to(torch.float32)
    return tensor.reshape(tensor.shape[0], -1)[None]


def _logits(value, frames):
    import torch

    tensor = value.detach().cpu() if hasattr(value, "detach") else torch.as_tensor(np.asarray(value))
    tensor = tensor.to(torch.float32).reshape(int(frames), 6)[None]
    return tensor


def _intervals(mask):
    edges = np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def _fill_gaps(mask, gap):
    """填掉内部短暂离开（两侧都在接触），不碰首尾。"""
    if gap <= 0:
        return mask
    out = mask.copy()
    length = len(out)
    index = 0
    while index < length:
        if out[index]:
            index += 1
            continue
        end = index
        while end < length and not out[end]:
            end += 1
        if index > 0 and end < length and end - index <= gap:
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
    """(T, 2, 2) XZ → (T, 2) m/s；第一帧复用第一段差分。"""
    if len(points) <= 1:
        return np.zeros((len(points), points.shape[1]), dtype=np.float32)
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


def _normalize_ranges(ranges, frames):
    """[[ [s,e), ...], ...] → 排序、合并、裁剪到 [0, frames]。"""
    result = []
    for track in ranges or [[], []]:
        normalized = []
        for pair in sorted(track):
            if len(pair) != 2:
                continue
            start, end = int(pair[0]), int(pair[1])
            start = max(0, min(frames, start))
            end = max(0, min(frames, end))
            if start >= end:
                continue
            if normalized and start <= normalized[-1][1]:
                normalized[-1] = [normalized[-1][0], max(normalized[-1][1], end)]
            else:
                normalized.append([start, end])
        result.append(normalized)
    while len(result) < 2:
        result.append([])
    return result[:2]


def _mask_from_ranges(ranges, frames):
    mask = np.zeros((frames, 2), dtype=np.uint8)
    for foot, track in enumerate(ranges[:2]):
        for start, end in track:
            mask[start:end, foot] = 1
    return mask


class ZMocapFootContact:
    """从 ExMR motion 检测左右脚接触，输出 contact 数据（不修改 motion）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"motion": ("ZMOCAP_MOTION",)},
            "optional": {
                "static_conf": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "height_tolerance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005}),
                "speed_limit": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.05}),
                "vertical_limit": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.05}),
                "min_frames": ("INT", {"default": 2, "min": 1, "max": 240, "step": 1}),
                "gap_frames": ("INT", {"default": 2, "min": 0, "max": 240, "step": 1}),
            },
        }

    RETURN_TYPES = ("ZMOCAP_MOTION", "ZMOCAP_CONTACTS")
    RETURN_NAMES = ("motion", "contacts")
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "ExMR 的 static_conf + SMPL-X 脚底高度 → 左右脚接触区间（zmocap.contact.v1）。"

    def run(self, motion, static_conf=0.8, height_tolerance=0.05, speed_limit=0.7,
            vertical_limit=0.8, min_frames=2, gap_frames=2):
        skeleton = str(motion.get("skeleton") or ("smplx" if "params" in motion else ""))
        if skeleton != "smplx":
            raise ValueError(f"Foot Contact 需要 SMPL-X motion，收到 {skeleton or '未知'}")
        logits = motion.get("static_conf_logits")
        if logits is None:
            raise ValueError("motion 缺少 static_conf_logits；请用本仓库的 ZMocapExMR 输出（或重新导出）")

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
        conf = _logits(logits, frames)

        with torch.inference_mode():
            # (1, T, 3): 左脚最低点、右脚最低点、全身最低点
            heights = lowest_vertex_height(
                params, body,
                [body.left_foot_vertices, body.right_foot_vertices, slice(None)],
            )[0].numpy()

        foot_height = heights[:, :2].astype(np.float32)
        body_lowest = heights[:, 2:3]
        probability = torch.sigmoid(conf)[0].numpy()[:, :4]
        static = np.stack([probability[:, 0:2].max(1), probability[:, 2:4].max(1)], axis=1)

        contact = (static > float(static_conf)) & (foot_height <= body_lowest + float(height_tolerance))

        # 速度门限：踝/脚关节的 XZ 速度 + 脚底垂直速度；降低“低但在滑”的误判。
        joints = motion.get("joints")
        if joints is not None and np.asarray(joints).ndim == 3 and np.asarray(joints).shape[1] >= 11:
            joints = np.asarray(joints, dtype=np.float32)
            left = (joints[:, 7] + joints[:, 10]) * 0.5
            right = (joints[:, 8] + joints[:, 11]) * 0.5
            points = np.stack([left[:, [0, 2]], right[:, [0, 2]]], axis=1)  # (T, 2, 2)
            speed = _speed(points, motion.get("fps", 30.0))
            vertical = _vertical_speed(foot_height, motion.get("fps", 30.0))
            contact &= (speed < float(speed_limit)) & (vertical < float(vertical_limit))

        for foot in range(2):
            contact[:, foot] = _drop_short(
                _fill_gaps(contact[:, foot], int(gap_frames)), int(min_frames)
            )

        floor = float(foot_height.min())
        ranges = [_intervals(contact[:, foot]) for foot in range(2)]
        feet = {}
        for foot, name in enumerate(("left", "right")):
            events = []
            for start, end in ranges[foot]:
                events.append({"frame": int(start), "type": "contact_start"})
                events.append({"frame": int(end), "type": "contact_end"})
            feet[name] = {"ranges": ranges[foot], "events": events}

        contacts = {
            "format": CONTACT_FORMAT,
            "fps": float(motion.get("fps", 30.0)),
            "frames": int(frames),
            "space": {"frame": "world", "up": "Y", "unit": "m"},
            "floor": {"mode": "plane", "height": floor},
            "feet": feet,
            "arrays": {
                "contact": contact.astype(np.uint8),
                "probability": static.astype(np.float32),
                "height": foot_height,
            },
            "meta": {
                "source": "auto",
                "algorithm": "exmr_static_conf+foot_height",
                "thresholds": {
                    "static_conf": float(static_conf),
                    "height_tolerance": float(height_tolerance),
                    "speed_limit": float(speed_limit),
                    "vertical_limit": float(vertical_limit),
                    "min_frames": int(min_frames),
                    "gap_frames": int(gap_frames),
                },
                "version": CONTACT_VERSION,
                "motion_sha256": _motion_hash(motion),
            },
        }
        left = sum(end - start for start, end in ranges[0])
        right = sum(end - start for start, end in ranges[1])
        print(f"[zmocap] foot contact: {frames} 帧，左 {len(ranges[0])} 段/{left} 帧，"
              f"右 {len(ranges[1])} 段/{right} 帧，floor {floor:.3f} m")
        return (motion, contacts)


def validate_contacts(contacts, motion) -> list[str]:
    """按 ``zmocap.contact.v1`` 校验；返回警告列表（不致命的问题）。"""
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
    if contact.ndim != 2 or contact.shape[1] != 2:
        raise ValueError(f"arrays.contact 形状应为 (T,2)，实际 {contact.shape}")

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
    """把触地数据变成预览用的双层光圈：每帧左右脚的着地位置与开关。"""
    if "joints" not in motion:
        raise ValueError("触地光圈需要带关节序列的 SMPL-X 运动（VRM 运动只有旋转，无法定位地面点）")
    warnings = validate_contacts(contacts, motion)
    joints = np.asarray(motion["joints"], dtype=np.float64)
    contact = np.asarray((contacts.get("arrays") or {})["contact"]) > 0
    frames = min(joints.shape[0], contact.shape[0])
    floor = float((contacts.get("floor") or {}).get("height", 0.0))

    def side(column: int, foot_joint: int, ankle_joint: int) -> dict:
        points = np.stack([joints[:frames, foot_joint], joints[:frames, ankle_joint]], axis=1).mean(axis=1)
        position = np.stack([points[:, 0], np.full(frames, floor), points[:, 2]], axis=1)
        return {
            "on": contact[:frames, column].astype(int).tolist(),
            "pos": np.round(position, 4).tolist(),
        }

    return {
        "format": CONTACT_FORMAT,
        "fps": float(contacts.get("fps", motion.get("fps", 30.0))),
        "frames": int(frames),
        "floor": floor,
        "left": side(0, 10, 7),
        "right": side(1, 11, 8),
        "warnings": warnings,
    }


class ZMocapContactLoad:
    """读取手改后的 ``zmocap.contact.v1`` JSON，覆盖自动区间。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"path": ("STRING", {"default": ""})},
            "optional": {"motion": ("ZMOCAP_MOTION",)},
        }

    RETURN_TYPES = ("ZMOCAP_CONTACTS",)
    RETURN_NAMES = ("contacts",)
    FUNCTION = "run"
    CATEGORY = "zmocap"
    DESCRIPTION = "读取手改的 contacts.json（source=manual），用于覆盖自动检测的区间。"

    def _resolve(self, path):
        candidate = Path(str(path)).expanduser()
        if candidate.is_file():
            return candidate
        try:
            import folder_paths

            for root in (Path(folder_paths.get_output_directory()),
                         Path(folder_paths.get_input_directory())):
                if (root / candidate).is_file():
                    return root / candidate
        except Exception:
            pass
        raise FileNotFoundError(f"contacts JSON 不存在：{path}")

    def run(self, path, motion=None):
        file = self._resolve(path)
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("format") != CONTACT_FORMAT:
            raise ValueError(f"不是 {CONTACT_FORMAT}：{file}")
        frames = int(data.get("frames", 0))
        if frames <= 0:
            raise ValueError("contacts JSON 缺少 frames")
        ranges = _normalize_ranges(
            [data.get("feet", {}).get("left", {}).get("ranges", []),
             data.get("feet", {}).get("right", {}).get("ranges", [])],
            frames,
        )
        if motion is not None:
            actual = int(np.asarray(motion.get("joints")).shape[0]) if motion.get("joints") is not None else frames
            if actual != frames:
                print(f"[zmocap] contacts frames={frames} 与 motion frames={actual} 不一致，请确认")
            expected = (data.get("meta") or {}).get("motion_sha256")
            if expected and motion.get("params") is not None:
                actual_hash = _motion_hash(motion)
                if expected != actual_hash:
                    print("[zmocap] contacts 与当前 motion 的 sha256 不一致（手动覆盖，继续）")
        contact = _mask_from_ranges(ranges, frames)
        feet = {}
        for foot, name in enumerate(("left", "right")):
            events = []
            for start, end in ranges[foot]:
                events.append({"frame": int(start), "type": "contact_start"})
                events.append({"frame": int(end), "type": "contact_end"})
            feet[name] = {"ranges": ranges[foot], "events": events}
        meta = dict(data.get("meta") or {})
        meta.update({"source": "manual", "version": CONTACT_VERSION})
        contacts = {
            "format": CONTACT_FORMAT,
            "fps": float(data.get("fps", 30.0)),
            "frames": frames,
            "space": data.get("space") or {"frame": "world", "up": "Y", "unit": "m"},
            "floor": data.get("floor") or {"mode": "plane", "height": 0.0},
            "feet": feet,
            "arrays": {"contact": contact, "probability": None, "height": None},
            "meta": meta,
        }
        print(f"[zmocap] contact load: {file.name}，左 {len(ranges[0])} 段，右 {len(ranges[1])} 段")
        return (contacts,)
