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
                "motion": ("ZMOCAP_MOTION",),
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
            bvh_path.write_text(_bvh(joints, float(motion.get("fps", 30.0))), encoding="utf-8")
            paths.append(str(bvh_path))
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


SMPLX_PARENTS = [
    0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,  # body
    15, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,   # face
    20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
]


def _bvh(joints: np.ndarray, fps: float) -> str:
    """最小 BVH 写出：以 SMPL-X 关节树为骨架，通道填 0（关节位置本身就是动画）。"""
    count, joint_count, _ = joints.shape
    names = [f"J{i}" for i in range(joint_count)]
    rest = joints[0]
    lines = ["HIERARCHY", f"ROOT {names[0]}", "{"]
    lines.append(f'  OFFSET 0 0 0')
    lines.append("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    children = {i: [] for i in range(joint_count)}
    for index, parent in enumerate(SMPLX_PARENTS):
        if index == 0:
            continue
        children[parent].append(index)

    def emit(index, depth):
        pad = "  " * depth
        for child in children[index]:
            offset = rest[child] - rest[index]
            lines.append(f"{pad}JOINT {names[child]}")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
            emit(child, depth + 1)
            lines.append(f"{pad}}}")

    emit(0, 1)
    lines.append("}")
    lines.append("MOTION")
    lines.append(f"Frames: {count}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")
    for frame in joints:
        root = frame[0]
        lines.append(" ".join(f"{value:.6f}" for value in [root[0], root[1], root[2], 0, 0, 0]))
    return "\n".join(lines) + "\n"
