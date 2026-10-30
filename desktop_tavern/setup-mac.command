#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "此安装入口支持 Apple Silicon Mac；Intel Mac 的 PyTorch 依赖尚未验证。"
  exit 2
fi
if [[ ! -x .venv-desktop/bin/python ]]; then
  tavern_python="${TAVERN_PYTHON:-}"
  if [[ -z "$tavern_python" ]]; then
    for candidate in python3 python3.10 python3.11 python3.12 python3.13; do
      if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        tavern_python="$(command -v "$candidate")"
        break
      fi
    done
  fi
  if [[ -z "$tavern_python" ]]; then
    echo "请安装 Python 3.10+，或用 TAVERN_PYTHON 指定路径。"
    exit 2
  fi
  "$tavern_python" -m venv .venv-desktop
fi
.venv-desktop/bin/python -m pip install -r desktop_tavern/requirements.txt
if [[ ! -f desktop_tavern/config.mac.local.json ]]; then
  cp desktop_tavern/config.mac.example.json desktop_tavern/config.mac.local.json
fi
echo "安装完成。请阅读 docs/DESKTOP_V2.md 配置玩家名、日志和校准。"
echo "双击 doctor-mac.command 检查；start-mac.command 默认只输出建议。"
