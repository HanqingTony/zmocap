"""MotionClip 契约（``zmocap.motion/1``）。

所有跨节点的动作数据都必须是一个带自描述头的 dict：

::

    {
      "format": "zmocap.motion/1",
      "skeleton": "smplx" | "soma" | "vrm",
      "fps": 30.0,
      "frames": 134,
      "joints":   (T, J, 3) float32         # 世界坐标，米，Y-up，正面 -Z
      "params":   {...}                     # 仅 smplx：SMPL-X 参数
      "world_rotation": (T, B, 3, 3)        # 仅 soma/vrm：世界旋转
      "names": [str, ...], "parents": [int, ...],
      "digest": "sha256:...",               # 内容指纹
      "sources": {...}                      # 来源链（调试/追溯）
    }

`digest` 覆盖几何与采样率（骨架、帧率、帧数、关节名、关节坐标、关键参数），
因此"把 SOMA 产物接到 SMPL 端口"这类接线错误会在解码处立刻报错，而不是渲染出怪东西。
"""

from __future__ import annotations

import hashlib

import numpy as np

FORMAT = "zmocap.motion/1"
SKELETONS = {
    "smplx": 55,      # SMPL-X 55 关节（含手指/脸）
    "soma": 78,       # SOMA 78 关节（通用交换格式）
    "vrm": 52,        # 标准 VRM humanoid 52 骨
}
COORDINATE_SYSTEM = "y-up-right-handed-front-minus-z"
UNITS = "meter"

# 参与 digest 的字段（顺序固定，避免浮点抖动）
_DIGEST_FIELDS = ("joints", "world_rotation", "transl", "static_conf_logits")
_DIGEST_PARAMS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl", "betas")


class MotionError(ValueError):
    """MOTION 契约被破坏。"""


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _feed(hasher: hashlib._Hash, name: str, array) -> None:
    data = _to_numpy(array)
    if data is None:
        return
    # 固定到 1e-6 量化后做 float32 序列化，避免不同计算路径的末位抖动
    quantized = np.round(data.astype(np.float64), 6).astype(np.float32)
    hasher.update(name.encode())
    hasher.update(str(quantized.shape).encode())
    hasher.update(np.ascontiguousarray(quantized).tobytes())


def compute_digest(clip: dict) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(clip.get("skeleton", "")).encode())
    hasher.update(str(clip.get("format", FORMAT)).encode())
    hasher.update(f"{float(clip.get('fps', 0.0)):.6f}".encode())
    hasher.update(str(int(clip.get("frames", 0))).encode())
    names = clip.get("names")
    if names is not None:
        for name in names:
            hasher.update(str(name).encode())
    parents = clip.get("parents")
    if parents is not None:
        for parent in parents:
            if parent is None:
                hasher.update(b"root")
            elif isinstance(parent, str):
                hasher.update(b"p:" + parent.encode())
            else:
                hasher.update(int(parent).to_bytes(4, "little", signed=True))
    for field in _DIGEST_FIELDS:
        _feed(hasher, field, clip.get(field))
    params = clip.get("params") or {}
    for field in _DIGEST_PARAMS:
        _feed(hasher, f"params.{field}", params.get(field))
    return "sha256:" + hasher.hexdigest()


def make_clip(clip: dict, skeleton: str, *, sources: dict | None = None, digest: bool = True) -> dict:
    """补齐契约字段并计算 fingerprint。"""
    if skeleton not in SKELETONS:
        raise MotionError(f"未知骨架类型 {skeleton!r}，可选：{sorted(SKELETONS)}")
    clip = dict(clip)
    clip["format"] = FORMAT
    clip["skeleton"] = skeleton
    clip["coordinate_system"] = COORDINATE_SYSTEM
    clip["units"] = UNITS
    joints = clip.get("joints")
    if joints is not None:
        clip["joints"] = _to_numpy(joints).astype(np.float32)
        clip["frames"] = int(clip["joints"].shape[0])
    clip["fps"] = float(clip.get("fps", 30.0))
    if sources is not None:
        clip["sources"] = dict(sources)
    elif "sources" not in clip:
        clip["sources"] = {}
    if digest:
        clip["digest"] = compute_digest(clip)
    return clip


def validate_clip(clip, skeleton: str | None = None, *, strict_digest: bool = True) -> dict:
    """校验契约；返回摘要字符串供节点显示。不合法直接抛 MotionError。"""
    if not isinstance(clip, dict):
        raise MotionError(f"motion 必须是 dict，收到 {type(clip).__name__}")
    fmt = clip.get("format")
    if fmt != FORMAT:
        raise MotionError(f"motion 契约版本不匹配：{fmt!r}（此节点只接受 {FORMAT!r}）")
    found = clip.get("skeleton")
    if found not in SKELETONS:
        raise MotionError(f"缺少/非法 skeleton 字段：{found!r}")
    if skeleton is not None and found != skeleton:
        raise MotionError(
            f"类型不匹配：此输入需要 skeleton={skeleton!r}，实际是 {found!r}。"
            f"{'请先加一个 zmocap SOMA2VRM 节点做格式转换。' if skeleton == 'vrm' and found == 'soma' else ''}"
        )
    frames = int(clip.get("frames") or 0)
    if frames <= 0:
        raise MotionError("motion 没有帧（frames<=0）")
    if skeleton and SKELETONS.get(found):
        expected = SKELETONS[found]
        names = clip.get("names")
        if names is not None and len(names) != expected and clip.get("joints") is not None and clip["joints"].shape[1] != expected:
            raise MotionError(f"{found} 骨架应有 {expected} 个关节，实际 joint 维度 {clip['joints'].shape[1]}")
    if strict_digest:
        stored = clip.get("digest")
        if not stored:
            raise MotionError("motion 缺少 digest，请用 make_clip() 生成")
        actual = compute_digest(clip)
        if actual != stored:
            raise MotionError(f"motion digest 不一致（数据被修改或序列化有损）：{stored} != {actual}")
    fps = float(clip.get("fps", 0.0))
    return f"{found} · {frames} 帧 · {fps:.2f} fps"


def stamp(clip: dict) -> str:
    """节点 UI 上显示用的简短摘要。"""
    digest = str(clip.get("digest", ""))[7:19]
    return f"{clip.get('skeleton')} {clip.get('frames')}f {clip.get('fps'):.0f}fps #{digest}"
