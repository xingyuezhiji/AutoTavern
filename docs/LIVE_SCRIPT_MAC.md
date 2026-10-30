# Mac 酒馆战棋模型脚本

这个版本读取 Mac 炉石的 Power.log，用现有训练权重给出操作建议，也可执行已校准的鼠标动作。当前安装流程已在 Apple Silicon Mac 验证，使用 CPU 推理，默认只输出建议。Intel Mac 的 PyTorch 二进制依赖尚未适配。

目前已经完成离线链路和 Mac 后端验证，尚未在正式炉石客户端实测。动作范围和日志缺失项见 [共用说明](LIVE_SCRIPT.md)。模型仍使用 4017 个数值特征、447 个卡牌编号和 245 个动作；包内权重为 917730 的 2,992,951 参数模型。

## 安装和配置

将整个便携包解压，双击 `live_tavern/setup-mac.command`。需要 Python 3.10 或更新版本和网络连接，脚本会查找已安装的兼容 Python，再安装 PyTorch、NumPy 和 PyObjC。源码仓库不含权重；便携包包含 `models/tavern_bc_policy.pt`。

如果解压工具未保留执行权限，在终端进入解压目录运行：

```bash
chmod +x live_tavern/*.command
./live_tavern/setup-mac.command
```

编辑安装后生成的 `live_tavern/config.mac.local.json`：

- `player_name` 填日志中的玩家名，包含实际的 `#` 后缀。
- `log_root` 默认 `/Applications/Hearthstone/Logs`。自定义安装路径时填真实目录，也可填完整的 `Power.log` 路径，支持 `~`。
- `window_owner`、`window_title` 默认 `Hearthstone`，必须与当前窗口完全一致。检查入口会列出匹配应用的窗口标题。
- `accepted_builds` 默认为空。核验日志与卡库兼容性后才填写 BuildNumber；当前训练卡库版本为 `36.6.3.253932`。

在 `~/Library/Preferences/Blizzard/Hearthstone/log.config` 中启用 Power 日志，编辑前保留原有其他配置，修改后重启炉石：

```ini
[Power]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false
Verbose=true
```

双击 `doctor-mac.command` 检查日志路径、窗口与权限。这个检查不会发送鼠标事件或申请系统权限。要校准或执行，在“系统设置 → 隐私与安全性”中为启动脚本的 Terminal 或 IDE 开启“辅助功能”“输入监控”和“屏幕与系统音频录制”（部分 macOS 版本称“屏幕录制”），然后重启 Terminal 或 IDE。屏幕录制权限用于获取完整窗口元信息，脚本不保存屏幕画面。建议模式和离线回放不需要这些桌面权限。

## 校准实际位置

Mac 坐标按窗口外框的逻辑点记录，包含标题栏；Quartz 鼠标事件使用相同坐标空间。Retina 显示器不额外乘以 2。不要复用 Windows 校准或包内合成样例。调整窗口大小、切换全屏或显示器缩放后重新校准。

双击 `calibrate-mac.command`，按提示选按钮或一排卡牌，也可以直接运行：

```bash
./live_tavern/calibrate-mac.command --point hand_drop
./live_tavern/calibrate-mac.command --row shop --count 3
```

命令启动后切到炉石，将鼠标放在目标位置，按 F6 记录；部分 Mac 键盘需要 Fn+F6。每种卡牌数量单独记录，从左到右逐个按键。商店、手牌、我方场面、插入空隙、发现选项分别为 `shop`、`hand`、`board`、`board_gaps`、`choice`，按钮名称见共用说明。未校准的动作不会参与模型选择。

`client_size` 表示 Mac 窗口外框的宽高，名字沿用共用配置。尺寸变化时，使用新的空白 Mac 示例配置重新记录，避免旧点位混用。

## 启动与停止

手动进入单人酒馆战棋并完成英雄选择，双击 `start-mac.command` 查看建议。日志输出写入 `runs/live/mac-decisions.jsonl`。源码运行可指定本地权重：

```bash
TAVERN_CHECKPOINT="/实际路径/tavern_bc_policy.pt" ./live_tavern/start-mac.command
```

校准、权限和客户端版本检查完成后，在终端明确开启实验性执行：

```bash
./live_tavern/start-mac.command --execute --allow-estimated-context
```

脚本只操作前台最前面的匹配窗口。一次只执行一个动作，并等待日志确认；拖动途中失焦或移动窗口时中止并松开鼠标。确认超时、状态过期或每回合动作上限也会停止。日志仍缺少部分效果和历史状态，`--allow-estimated-context` 接受这项限制，核心信息缺失仍会阻止动作。

按 F8（或 Fn+F8）、Ctrl+C，或双击 `stop-mac.command` 停止。执行时把鼠标移到炉石窗口外框左上角也会停止。STOP 文件保留，确认重新运行后双击 `reset-mac.command` 清除。脚本不会自动匹配、选英雄或处理断线重连。

## 离线验证

便携包中的合成日志可验证模型链路，不会操作鼠标：

```bash
.venv-live/bin/python -m live_tavern \
  --config samples/geometry.json \
  --checkpoint models/tavern_bc_policy.pt replay samples/Power.log
```

检查权重与卡库：

```bash
.venv-live/bin/python -m live_tavern \
  --checkpoint models/tavern_bc_policy.pt doctor
```

Mac 后端验证结果见 `docs/mac-live-script-verification.json`；共用模型和原生模拟器结果见 `docs/live-script-verification.json`。后端使用模拟 Quartz 事件验证失焦、移动窗口、点击拖动及释放行为，真实系统只读检查不发送事件。这些结果尚不能证明正式客户端的操作准确率或对局胜率。
