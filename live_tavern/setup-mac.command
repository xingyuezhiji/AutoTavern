#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin ]]; then
  echo "此脚本需要 macOS。"
  exit 2
fi
if [[ ! -x .venv-live/bin/python ]]; then
  tavern_python="${TAVERN_PYTHON:-}"
  if [[ -z "$tavern_python" ]]; then
    for candidate in python3 python3.10 python3.11 python3.12 python3.13; do
      if command -v "$candidate" >/dev/null && \
         "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        tavern_python="$(command -v "$candidate")"
        break
      fi
    done
  fi
  if [[ -z "$tavern_python" ]]; then
    echo "请先安装 Python 3.10 或更新版本；也可用 TAVERN_PYTHON 指定 Python 路径。"
    exit 2
  fi
  "$tavern_python" -c 'import sys; assert sys.version_info >= (3, 10), "需要 Python 3.10 或更新版本"'
  "$tavern_python" -m venv .venv-live
fi
.venv-live/bin/python -m pip install -r live_tavern/requirements.txt
if [[ ! -f live_tavern/config.mac.local.json ]]; then
  cp live_tavern/config.mac.example.json live_tavern/config.mac.local.json
fi
echo "安装完成。请按 docs/LIVE_SCRIPT_MAC.md 配置玩家名、日志和坐标。"
echo "先运行 doctor-mac.command 检查，再运行 start-mac.command 查看建议。"
