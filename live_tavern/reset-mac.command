#!/bin/bash
set -euo pipefail
rm -f "$(dirname "$0")/STOP"
echo "已清除停止标记。可重新启动。"
