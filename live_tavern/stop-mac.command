#!/bin/bash
set -euo pipefail
touch "$(dirname "$0")/STOP"
echo "停止标记已写入；当前脚本会退出。"
