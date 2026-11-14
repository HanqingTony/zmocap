"""VRM 链路冒烟：SMPL-X → SOMA → VRM → Calibrate → Bind。

用法（在 ComfyUI 的 python 里执行，保证 folder_paths 可用）::

    ~/zvenv/comfyui/bin/python scripts/smoke_vrm_chain.py <exmr_results.pt> <character.vrm>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path.home() / "zrepo/ComfyUI"
for entry in (str(ROOT.parent), str(COMFY)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zmocap.nodes.format_soma import ZMocapSMPL2SOMA, ZMocapSOMA2VRM  # noqa: E402
from zmocap.nodes.calibrate import ZMocapBind, ZMocapCalibrate  # noqa: E402
from zmocap.motion import make_clip  # noqa: E402
from zmocap.nodes.exmr import _joints  # noqa: E402

SMPLX = Path.home() / "zext/comfyui/models/zmocap/body_models/smplx/SMPLX_NEUTRAL.npz"


def _smplx_clip(payload: dict, fps: float) -> dict:
    params = {key: value for key, value in payload["smplx_global"].items()}
    joints = _joints(params, str(SMPLX))
    clip = {
        "params": params,
        "joints": joints,
        "fps": float(fps),
        "frames": int(joints.shape[0]),
        "static_conf_logits": payload.get("static_conf_logits"),
        "source": "exmr/sapiens2",
    }
    return make_clip(clip, "smplx", sources={"engine": "exmr", "keypoints": "sapiens2", "hands": "hamer"})


def main(take: str, character: str) -> int:
    payload = torch.load(take, map_location="cpu", weights_only=False)
    motion = _smplx_clip(payload, 30.0)
    print("1) SMPL-X :", motion["digest"][7:19], motion["joints"].shape)

    (soma,) = ZMocapSMPL2SOMA().run(motion)
    print("2) SOMA   :", soma["digest"][7:19], np.asarray(soma["world"]).shape)

    (vrm,) = ZMocapSOMA2VRM().run(soma)
    print("3) VRM    :", vrm["digest"][7:19], np.asarray(vrm.get("rotation", vrm.get("joints"))).shape)

    calibration, cal_report = ZMocapCalibrate().run({"path": character, "label": Path(character).stem})
    print("4) 标定   :", cal_report)

    glb, bind_report = ZMocapBind().run(vrm, calibration, "zmocap_smoke", True)
    print("5) 绑定   :", glb["relative"], glb["scale"], bind_report)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
