#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin ]]; then
  echo "此脚本需要 macOS。"
  exit 2
fi
if [[ ! -x .venv-live/bin/python || ! -f live_tavern/config.mac.local.json ]]; then
  echo "请先运行 live_tavern/setup-mac.command。"
  exit 2
fi
if [[ -f live_tavern/STOP && "$1" != ui-doctor ]]; then
  echo "已停止。确认准备重新运行后，双击 reset-mac.command 清除停止标记。"
  exit 2
fi
mode="$1"
shift
cli_args=(--config live_tavern/config.mac.local.json)
if [[ "$mode" == watch ]]; then
  checkpoint="${TAVERN_CHECKPOINT:-models/tavern_bc_policy.pt}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "找不到权重：$checkpoint"
    echo "请设置 TAVERN_CHECKPOINT 为实际文件路径，或使用含权重的便携包。"
    exit 2
  fi
  cli_args+=(--checkpoint "$checkpoint" --output runs/live/mac-decisions.jsonl)
fi
exec .venv-live/bin/python -m live_tavern "${cli_args[@]}" "$mode" "$@"
