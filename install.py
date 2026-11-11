#!/usr/bin/env python3
"""模型自检：确认 ComfyUI/models/zmocap 下的权重是否齐全。

用法: python install.py [--models-dir PATH]
"""

import argparse
from pathlib import Path

# 权重清单（相对 models/zmocap），以 ExMR 推理链为准
REQUIRED = {
    "exmr/exmr.safetensors": 1_500_000_000,
    "sapiens2/sapiens2_1b_pose.safetensors": 4_000_000_000,
    "body_models/smplx/SMPLX_NEUTRAL.npz": 1_000_000,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=None, help="默认 <ComfyUI>/models/zmocap")
    parser.add_argument("--vendor-dir", default=None, help="默认 <仓库>/vendor/ExMR")
    args = parser.parse_args()
    root = Path(args.models_dir) if args.models_dir else Path(__file__).parent.parent.parent / "models" / "zmocap"
    print(f"模型目录: {root}")
    missing = 0
    for rel, minimum in REQUIRED.items():
        path = root / rel
        ok = path.is_file() and path.stat().st_size >= minimum
        print(f"  {'OK  ' if ok else 'MISS'} {rel}")
        missing += 0 if ok else 1

    vendor = Path(args.vendor_dir) if args.vendor_dir else Path(__file__).parent / "vendor" / "ExMR"
    if (vendor / "exmr").is_dir():
        import sys

        sys.path.insert(0, str(Path(__file__).parent / "support"))
        import vendor_patch

        changed = vendor_patch.patch(vendor)
        print(f"  vendor 助手: {len(changed)} 个文件已改写")
    else:
        print(f"  vendor 未就位（{vendor}），跳过助手注入")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
