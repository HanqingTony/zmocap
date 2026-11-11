#!/bin/bash
# dev-sync.sh — 把工作区同步到 zmain 的 ComfyUI（开发用，不算发布流程）
# 用法: bash scripts/dev-sync.sh [user@host]
# 说明: vendor/ 与 models/ 只在远端维护，不参与同步
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-tony@192.168.0.102}"
DEST="/home/tony/zrepo/zmocap/"
rsync -a --delete \
  --exclude '.git/' --exclude 'vendor/' --exclude 'models/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  ./ "$TARGET:$DEST"
echo "已同步到 $TARGET:$DEST"
