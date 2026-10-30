#!/bin/bash
set -euo pipefail
if [[ "$#" -eq 0 ]]; then
  echo "输入 point 校准按钮或落点；输入 row 校准一排卡牌。"
  read -r -p "类型（point/row）：" kind
  case "$kind" in
    point)
      echo "end refresh freeze upgrade power0 power1 neutral hand_drop sell_drop spell_drop"
      read -r -p "位置名称：" name
      set -- --point "$name"
      ;;
    row)
      echo "board board_gaps hand shop choice"
      read -r -p "行名称：" name
      read -r -p "当前槽位数量：" count
      set -- --row "$name" --count "$count"
      ;;
    *) echo "类型必须是 point 或 row。"; exit 2 ;;
  esac
fi
exec /bin/bash "$(dirname "$0")/run-mac.sh" calibrate "$@"
