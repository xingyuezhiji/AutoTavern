#!/bin/bash
set -euo pipefail
touch "$(dirname "$0")/STOP"
echo "已写入 STOP。需要重启时运行 reset-mac.command。"
