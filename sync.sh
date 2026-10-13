#!/usr/bin/env bash
# 一键同步本地改动到 GitHub (Zhaolarry/Info-VLA)
# 用法: ./sync.sh ["commit message"]
#
# 优先走 git push；如果本机到 github.com 的网络不通，自动降级为
# GitHub API 方式（需要提前 export GH_TOKEN=<你的 GitHub token>）。
set -euo pipefail
cd "$(dirname "$0")"

git add -A
if git diff --cached --quiet; then
    echo "没有新改动，无需提交"
else
    git commit -m "${1:-chore: update}"
fi

if git push origin main; then
    echo "✅ 已推送到 GitHub"
else
    echo "⚠️ git push 失败（网络或认证问题），尝试 GitHub API 方式..."
    if [ -z "${GH_TOKEN:-}" ]; then
        echo "请 export GH_TOKEN=<你的 GitHub token> 后重试"
        exit 1
    fi
    MSG=$(git log -1 --pretty=%B)
    python3 scripts/git_push_api.py "$GH_TOKEN" "$MSG"
    git fetch origin main
    git reset --hard origin/main
    echo "✅ 已通过 API 推送，本地已同步为远程快照"
fi
