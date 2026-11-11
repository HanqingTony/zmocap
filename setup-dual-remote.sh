#!/bin/bash
# setup-dual-remote.sh - 配置 GitHub + Gitee 双远端推送
# 用法: bash setup-dual-remote.sh   （在 zmocap 仓库根目录执行）
# 前置: Gitee 上已存在空仓库 HanqingTony/zmocap（Gitee 不支持纯 SSH 建仓）

set -euo pipefail
cd "$(dirname "$0")"

GH="git@github.com:HanqingTony/zmocap.git"
GT="git@gitee.com:HanqingTony/zmocap.git"

git config --unset-all remote.origin.pushurl 2>/dev/null || true
git remote set-url --add --push origin "$GH"
git remote set-url --add --push origin "$GT"

echo "双远端已配置："
git remote -v
echo ""
echo "现在执行 git push origin main 会同时推 GitHub + Gitee"
