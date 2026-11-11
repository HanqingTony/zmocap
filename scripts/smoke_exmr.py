#!/usr/bin/env python3
"""在 ComfyUI 之外直接跑 ExMR 三阶段，验证最小链路是否通。

用法:
  python scripts/smoke_exmr.py --frames FRAME_DIR --boxes tracking.npz --out out.npz
                               [--models MODELS_ROOT] [--fps 30]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nodes.exmr_engine import ExMRRunner, build_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--boxes", required=True, help="tracking.npz（含 boxes 与 K）")
    parser.add_argument("--out", default="exmr_smoke.npz")
    parser.add_argument("--models", default=str(Path.home() / "zext/comfyui/models/zmocap"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-hands", action="store_true")
    args = parser.parse_args()

    frames = Path(args.frames)
    assets = build_assets(Path(args.models), REPO / "vendor/ExMR")
    print("assets:", assets)

    with np.load(args.boxes, allow_pickle=True) as data:
        boxes = np.asarray(data["boxes"], dtype=np.float32)
        intrinsics = np.asarray(data["K"], dtype=np.float32)

    runner = ExMRRunner(assets)

    started = time.monotonic()
    keypoints = runner.keypoints(frames, boxes)
    print(f"keypoints {tuple(keypoints.shape)} in {time.monotonic() - started:.1f}s")

    started = time.monotonic()
    body, hands = runner.features(frames, boxes, intrinsics, keypoints=keypoints, hands=not args.no_hands)
    print(f"features body{tuple(body.shape)} in {time.monotonic() - started:.1f}s")

    data = runner.model_inputs(keypoints, boxes, intrinsics, body=body, hands=hands)

    started = time.monotonic()
    native = runner.infer(data, static_cam=True, fps=args.fps)
    print(f"exmr in {time.monotonic() - started:.1f}s")

    np.savez_compressed(args.out, **{f"global_{k}": v.numpy() if hasattr(v, "numpy") else v for k, v in native["smplx_global"].items()})
    params = native["smplx_global"]
    print("output keys:", sorted(params))
    for key, value in params.items():
        print(f"  {key}: {tuple(value.shape)}")
    print("saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
