#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-desktop/bin/python ]]; then
  echo "请先运行 desktop_tavern/setup-mac.command。"
  exit 2
fi
cli_args=(--config desktop_tavern/config.mac.local.json --checkpoint "${TAVERN_CHECKPOINT:-models/tavern_bc_policy.pt}")
exec .venv-desktop/bin/python -m desktop_tavern "${cli_args[@]}" "$@"
