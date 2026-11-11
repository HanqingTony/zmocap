"""SMPL-X（ExMR）参数 → SOMA 78 关节运动。

输出口径与 motionbridge 的 `motionbridge.soma.v1` 文件一致：names / parents / rest / world / fps。
SMPL-X 的关节朝向在 rest 姿态下就是单位阵，所以 rest 旋转全部取 I、rest 位置取 SMPL-X 模板
关节坐标；这样 world 旋转即"SOMA 相对 rest 的变化量"，正好是下游重定向需要的语义。
"""

from __future__ import annotations

import numpy as np

SOMA_NAMES = [
    "Root", "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]

SOMA_PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7,
    4, 12, 13, 14, 15, 16, 17, 18, 15, 20, 21, 22,
    23, 15, 25, 26, 27, 28, 15, 30, 31, 32, 33, 15,
    35, 36, 37, 38, 4, 40, 41, 42, 43, 44, 45, 46,
    43, 48, 49, 50, 51, 43, 53, 54, 55, 56, 43, 58,
    59, 60, 61, 43, 63, 64, 65, 66, 1, 68, 69, 70,
    71, 1, 73, 74, 75, 76,
]

# SOMA 关节 ← SMPL-X 来源
#   ("j", i)         直接取 SMPL-X 关节 i
#   ("mid", a, b)    取 a、b 的中点
#   ("ext", a, b, k) 从 a 沿 (a-b) 方向再延伸 k 倍 |a-b|
#   ("same", name)   复制另一个 SOMA 关节
_FINGERS = {
    # SMPL-X 的每指 3 个关节 → SOMA 的 4 个 + 末端
    "Index": (25, 26, 27),
    "Middle": (28, 29, 30),
    "Pinky": (31, 32, 33),
    "Ring": (34, 35, 36),
    "Thumb": (37, 38, 39),
}
_FINGERS_RIGHT = {
    "Index": (40, 41, 42),
    "Middle": (43, 44, 45),
    "Pinky": (46, 47, 48),
    "Ring": (49, 50, 51),
    "Thumb": (52, 53, 54),
}


def _source_table() -> dict[str, tuple]:
    table: dict[str, tuple] = {
        "Root": ("same", "Hips"),
        "Hips": ("j", 0),
        "Spine1": ("j", 3),
        "Spine2": ("j", 6),
        "Chest": ("j", 9),
        "Neck1": ("j", 12),
        "Neck2": ("mid", 12, 15),
        "Head": ("j", 15),
        "HeadEnd": ("ext", 15, 12, 0.45),
        "Jaw": ("j", 22),
        "LeftEye": ("j", 23),
        "RightEye": ("j", 24),
        "LeftShoulder": ("j", 13),
        "LeftArm": ("j", 16),
        "LeftForeArm": ("j", 18),
        "LeftHand": ("j", 20),
        "RightShoulder": ("j", 14),
        "RightArm": ("j", 17),
        "RightForeArm": ("j", 19),
        "RightHand": ("j", 21),
        "LeftLeg": ("j", 1),
        "LeftShin": ("j", 4),
        "LeftFoot": ("j", 7),
        "LeftToeBase": ("j", 10),
        "LeftToeEnd": ("ext", 10, 7, 0.55),
        "RightLeg": ("j", 2),
        "RightShin": ("j", 5),
        "RightFoot": ("j", 8),
        "RightToeBase": ("j", 11),
        "RightToeEnd": ("ext", 11, 8, 0.55),
    }
    for side, fingers in (("Left", _FINGERS), ("Right", _FINGERS_RIGHT)):
        for finger, (a, b, c) in fingers.items():
            name = f"{side}Hand{finger}"
            table[f"{name}1"] = ("j", a)
            table[f"{name}2"] = ("j", b)
            table[f"{name}3"] = ("j", c)
            if finger == "Thumb":
                table[f"{name}End"] = ("ext", c, b, 0.6)
            else:
                table[f"{name}4"] = ("ext", c, b, 0.7)
                table[f"{name}End"] = ("ext", c, b, 1.1)
    return table


SOURCE = _source_table()

# SMPL-X 是 Y 轴向上，重定向侧（Blender/MotionBridge 口径）是 Z 轴向上
_Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class SomaMotion:
    """motionbridge 口径的 SOMA 运动（只读容器）。"""

    def __init__(self, names, parents, rest, world, fps):
        self.names = list(names)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.rest = np.asarray(rest, dtype=np.float32)
        self.world = np.asarray(world, dtype=np.float32)
        self.fps = float(fps)
        self._index = {name: i for i, name in enumerate(self.names)}

    def index(self, name):
        return self._index[str(name)]

    @property
    def frames(self):
        return int(self.world.shape[0])

    @property
    def positions(self):
        return self.world[:, :, :3, 3]

    def save(self, path):
        np.savez_compressed(
            path,
            format=np.array("motionbridge.soma.v1"),
            names=np.array(self.names),
            parents=self.parents,
            rest=self.rest,
            world=self.world,
            fps=np.float64(self.fps),
        )
        return path


def _soma_rest_positions(rest_joints: np.ndarray, table: dict) -> np.ndarray:
    """按来源表把 SMPL-X 模板关节坐标换算成 SOMA 的 rest 位置。"""
    out = np.zeros((len(SOMA_NAMES), 3), dtype=np.float64)
    known: dict[str, np.ndarray] = {}
    order = [i for i, n in enumerate(SOMA_NAMES) if table[n][0] != "same"]
    order += [i for i, n in enumerate(SOMA_NAMES) if table[n][0] == "same"]
    for index in order:
        name = SOMA_NAMES[index]
        rule = table[name]
        kind = rule[0]
        if kind == "j":
            position = rest_joints[rule[1]]
        elif kind == "mid":
            position = (rest_joints[rule[1]] + rest_joints[rule[2]]) / 2.0
        elif kind == "ext":
            a, b, scale = rest_joints[rule[1]], rest_joints[rule[2]], rule[3]
            position = a + (a - b) * scale
        else:  # same
            position = known[rule[1]]
        out[index] = position
        known[name] = position
    return out


def _smplx_rest_joints():
    from ..nodes.loader import models_root

    path = models_root() / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not path.is_file():
        raise FileNotFoundError(f"缺 SMPL-X 模板：{path}")
    data = np.load(str(path), allow_pickle=True)
    template = np.asarray(data["v_template"], dtype=np.float64)
    regressor = data["J_regressor"]
    joints = np.asarray(regressor.dot(template) if hasattr(regressor, "dot") else regressor, dtype=np.float64)
    return joints[:55]


def build_soma_motion(params, fps: float = 30.0) -> SomaMotion:
    """ExMR 的 SMPL-X 参数 → SomaMotion。"""
    from ..nodes.retarget import SMPLX_PARENTS, _mat_from_quat, local_rotations

    quats = local_rotations(params)                     # (T, 55, 4) xyzw
    frames = quats.shape[0]
    rest_joints = _smplx_rest_joints()

    transl = params.get("transl")
    transl = np.asarray(transl.detach().cpu() if hasattr(transl, "detach") else transl, dtype=np.float64)
    transl = transl.reshape(frames, 3)

    world_rot = np.zeros((frames, 55, 3, 3), dtype=np.float64)
    world_pos = np.zeros((frames, 55, 3), dtype=np.float64)
    for index in range(55):
        local = _mat_from_quat(quats[:, index, :])
        parent = SMPLX_PARENTS[index]
        if parent < 0:
            world_rot[:, index] = local
            world_pos[:, index] = rest_joints[index] + transl
        else:
            world_rot[:, index] = world_rot[:, parent] @ local
            offset = rest_joints[index] - rest_joints[parent]
            world_pos[:, index] = world_pos[:, parent] + world_rot[:, parent] @ offset

    table = SOURCE
    rest_soma = _soma_rest_positions(rest_joints, table)
    soma_index = {name: i for i, name in enumerate(SOMA_NAMES)}
    soma_rot = np.zeros((frames, len(SOMA_NAMES), 3, 3), dtype=np.float64)
    soma_pos = np.zeros((frames, len(SOMA_NAMES), 3), dtype=np.float64)

    order = [i for i, n in enumerate(SOMA_NAMES) if table[n][0] != "same"]
    order += [i for i, n in enumerate(SOMA_NAMES) if table[n][0] == "same"]
    for index in order:
        name = SOMA_NAMES[index]
        rule = table[name]
        kind = rule[0]
        if kind == "j":
            source = rule[1]
            soma_rot[:, index] = world_rot[:, source]
            soma_pos[:, index] = world_pos[:, source]
        elif kind == "mid":
            a, b = rule[1], rule[2]
            soma_rot[:, index] = world_rot[:, b]
            soma_pos[:, index] = (world_pos[:, a] + world_pos[:, b]) / 2.0
        elif kind == "ext":
            a, b, scale = rule[1], rule[2], rule[3]
            local_offset = (rest_joints[a] - rest_joints[b]) * scale
            soma_rot[:, index] = world_rot[:, a]
            soma_pos[:, index] = world_pos[:, a] + world_rot[:, a] @ local_offset
        else:  # same
            other = soma_index[rule[1]]
            soma_rot[:, index] = soma_rot[:, other]
            soma_pos[:, index] = soma_pos[:, other]

    # 注意：保持 SOMA 原生 Y 轴向上；重定向侧的坐标系由 rigmap 从角色骨架自行探测。

    rest = np.tile(np.eye(4, dtype=np.float64), (len(SOMA_NAMES), 1, 1))
    rest[:, :3, 3] = rest_soma
    world = np.tile(np.eye(4, dtype=np.float64), (frames, len(SOMA_NAMES), 1, 1))
    world[:, :, :3, :3] = soma_rot
    world[:, :, :3, 3] = soma_pos
    return SomaMotion(SOMA_NAMES, SOMA_PARENTS, rest, world, fps)
