# Third-party notice

`vendor/motionbridge/` 摘自 [powerhouse90/motionbridge](https://github.com/powerhouse90/motionbridge)，
用于骨架识别（`rigmap.py`）与重定向求解（`retarget.py`）。

- 许可证：Apache License 2.0
- 版权：Copyright (c) motionbridge contributors
- 本地改动：`retarget.py` 中 `from motionbridge.core import contacts` 改为 `from . import contacts`

SOMA 关节名/父子表（78 关节）由该项目的 `h3_walk.npz` 参照实现，本项目在 `zmocap/soma/motion.py`
里重新表述为常量与来源表，以便离线生成。
