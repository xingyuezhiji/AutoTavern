# AutoTavern 桌面 V2：Mac 与 Windows

这是新的 `desktop_tavern` 版本，参考 [HsBgAI](https://github.com/kanshufan-1107/HsBgAI) 的阶段管理、英雄统计选择和动作确认设计。招募仍由本地训练模型决定。旧 `live_tavern` 入口继续可用，配置各自独立。

两平台共用策略与状态机：大厅 → 排队 → 英雄选择 → 招募 / 发现 → 战斗 → 结算。脚本一次只执行一个动作，收到对应日志或界面确认后才继续。找局、结算默认关闭，启动默认只输出建议。

## 安装

解压对应平台的 ZIP，保留完整目录。包内包含 `models/tavern_bc_policy.pt`：启智任务 917730 的 2,992,951 参数模型，输入为 4017 个数值特征、447 个卡牌索引、245 个动作掩码。推理在 CPU 上运行，不需要模型 API Key 或 GPU。

### Mac

当前安装入口支持 Apple Silicon，Python 3.10+。双击 `desktop_tavern/setup-mac.command` 安装。若系统 `python3` 太旧，可从包根目录执行：

```bash
TAVERN_PYTHON=/opt/homebrew/bin/python3.10 bash desktop_tavern/setup-mac.command
```

在“系统设置 → 隐私与安全性”中，给启动程序的 Terminal 或 IDE 开启辅助功能、输入监控、屏幕录制权限，并重新打开该程序。校准和截图需要这些权限。Intel Mac 的新版 PyTorch 二进制依赖未验证，不能直接按 Apple Silicon 包宣称可用。

### Windows

安装 64 位 Python 3.10+，确保 `py -3` 可用，再双击 `desktop_tavern/setup-windows.cmd`。Windows 使用前台客户区与 DPI 坐标；Mac 使用 Quartz 逻辑点坐标，两套校准不能互用。

安装依赖需要联网。从源码启动时，将权重放到 `models/tavern_bc_policy.pt`，或设置 `TAVERN_CHECKPOINT` 为实际路径；ZIP 已带权重。

## 配置日志与玩家

安装会生成 `desktop_tavern/config.mac.local.json` 或 `config.windows.local.json`。修改：

| 配置 | 用途 |
| --- | --- |
| `player_name` | 日志里的完整玩家名，包含实际 `#` 后缀 |
| `log_root` | 炉石的 `Logs` 目录，或具体 `Power.log` 路径 |
| `window_title` | 实际前台游戏窗口标题 |
| `accepted_builds` | 经过核验的客户端 BuildNumber 字符串列表 |
| `auto_queue` / `auto_settle` | 是否自动点击已校准的找局 / 结算界面；默认 `false` |
| `max_games` | 本次运行对局上限；默认 1 |
| `max_actions_per_turn` | 单回合动作上限；默认 60 |
| `action_timeout` | 动作确认超时秒数；默认 15 |
| `hero_stats` | 本地英雄统计文件相对配置目录的路径；默认空 |

设置实际游戏安装目录，不要照抄默认路径。Power.log 一般位于 `Logs/Hearthstone_*/Power.log`。Mac 日志配置通常为 `~/Library/Preferences/Blizzard/Hearthstone/log.config`；Windows 通常为 `%LOCALAPPDATA%\Blizzard\Hearthstone\log.config`。若尚未启用，配置：

```ini
[Power]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false
Verbose=true
```

重启客户端后检查日志是否增长。`BuildNumber` 必须从当前日志核实，再确认与训练卡库及规则兼容。仅添加 build 字符串不会让模型自动适配新补丁；本包训练卡库为 36.6.3.253932，未知卡牌会阻止推理。

先运行 `doctor-mac.command` 或 `doctor-windows.cmd`，核验模型、词表、日志位置和已有校准。Mac doctor 也会只读检查权限与窗口，不会激活游戏或点击。

## 校准点位和卡牌行

将游戏保持在固定大小、固定显示模式。校准时切回炉石，指向目标后按 F6；Mac 可能需要 Fn+F6，F8 / Fn+F8 停止。以下命令从包根目录执行。

Mac 示例：

```bash
bash desktop_tavern/calibrate-mac.command --point hand_drop
bash desktop_tavern/calibrate-mac.command --row heroes --count 4
bash desktop_tavern/calibrate-mac.command --row shop --count 3
bash desktop_tavern/calibrate-mac.command --row hand --count 1
bash desktop_tavern/calibrate-mac.command --row board_gaps --count 1
```

Windows 使用相同参数：

```bat
desktop_tavern\calibrate-windows.cmd --row heroes --count 4
desktop_tavern\calibrate-windows.cmd --row shop --count 3
```

英雄、商店、手牌、棋盘和发现选项从左到右记录。不同数量的行分别校准：三商品坐标不会代替四商品坐标。`board_gaps` 是随从插入位置，数量等于当前棋盘随从数加一；它与 `board` 卡牌中心不同。

点位包括 `end`、`refresh`、`freeze`、`upgrade`、`power0`、`power1`、`neutral`、`hand_drop`、`sell_drop`、`spell_drop`，以及游戏外的 `queue`、`settle`、`hero_confirm`。行包括 `heroes`、`shop`、`hand`、`board`、`board_gaps`、`choice`。未校准的动作不会进入可执行动作集。

修改窗口大小后，需另建空配置重新校准。Mac 用窗口逻辑点，Retina 截图会归一到同一坐标空间。

## 校准界面模板

游戏外动作和英雄选择必须有用户采集的模板。脚本不会下载模板或猜测按钮位置。依次校准 `heroes`、`hero_confirm`、`lobby`、`queue`、`result`：

```bash
bash desktop_tavern/calibrate-mac.command --scene heroes
bash desktop_tavern/calibrate-mac.command --scene hero_confirm
```

```bat
desktop_tavern\calibrate-windows.cmd --scene lobby
desktop_tavern\calibrate-windows.cmd --scene queue
desktop_tavern\calibrate-windows.cmd --scene result
```

每次按两次 F6，分别采集稳定界面区域的左上角、右下角。选择该界面独有的文字或图案区域，避开动画、头像变化和大面积纯色。第二次采集后截取当前窗口，模板保存在 `desktop_tavern/templates/`，只在本地使用。

模板按局部 RGB 平均误差匹配，默认阈值 0.035。多个界面同时命中时等待；没有命中时不执行依赖该界面的动作。点击前重新截图，核验模板仍匹配、窗口未移动、日志状态未变。阈值不能替代可靠校准，先用建议模式观察命中结果。

英雄选择按日志里的实际 `ZONE_POSITION` 定位，过滤 `BACON_LOCKED_MULLIGAN_HERO`。选卡后等待 `hero_confirm` 模板，再点击校准的确认按钮；收到 `SendChoices` 才继续。没有确认模板时不会盲点。

## 本地英雄统计

支持以下 JSON，`avg_placement` 为平均名次，越低越优先。可以使用卡牌 ID 或 DBF 编号；皮肤支持 `BACON_SKIN_PARENT_ID` 匹配。下面是格式示例，不是推荐数据：

```json
{
  "heroes": [
    {"card_id": "YOUR_HERO_CARD_ID", "avg_placement": 4.1},
    {"dbf_id": 12345, "avg_placement": 4.2}
  ]
}
```

把文件放在配置目录，并设置 `"hero_stats": "heroes.local.json"`。应自行确保统计的补丁、模式和分段适用。无匹配统计时，使用可见护甲排序，同护甲按左侧槽位优先，并在记录中明确标注回退依据。

## 启动与停止

双击 `start-mac.command` / `start-windows.cmd` 启动建议模式。建议模式只读取日志并输出动作，不移动鼠标；如需同时观察场景模板，添加 `--observe-scenes`，此时只截图，不点击。

完成校准和客户端核验后，实验性执行模式：

```bash
bash desktop_tavern/start-mac.command --execute --allow-estimated-context
```

```bat
desktop_tavern\start-windows.cmd --execute --allow-estimated-context
```

Power.log 无法完整还原模拟器的效果计数器、部分对手局面等特征，因此需要显式接受估算上下文。此参数并不表示实战效果已验收。

要衔接下一局，配置中同时启用 `auto_queue`、`auto_settle`，增加 `max_games`，并完成对应模板和点位校准。找局后须看到排队界面或新对局；结算后须看到大厅。每个衔接周期只提交一次找局，日志变化不能触发重复点击。超时停止，不自动重试可能已执行的动作。

保持炉石在前台。按 F8、Ctrl+C，或双击 `stop-*` 脚本停止。STOP 文件持续生效；重新启动前运行 `reset-*`。运行中的会话不能被 reset 清除停止标记。每个安装目录只能运行一个会话。

记录保存在 `runs/desktop-v2/<时间-会话>/`：`events.jsonl` 包含阶段、建议、动作和确认；每局另有 `snapshots.jsonl`。日志、模板和个人配置不进入安装包或 Git。

## 离线回放与验收范围

包内 `samples/Power.log`、`samples/geometry.json` 是合成样例，不能拿样例坐标操作真实客户端。安装后回放：

Mac：

```bash
.venv-desktop/bin/python -m desktop_tavern --config samples/geometry.json replay samples/Power.log
```

Windows：

```bat
.venv-desktop\Scripts\python.exe -m desktop_tavern --config samples\geometry.json replay samples\Power.log
```

完整离线验收：

```bash
.venv-desktop/bin/python tools/verify_desktop_v2.py --checkpoint models/tavern_bc_policy.pt --report verification.json
```

验收覆盖英雄锁定和物理槽位、皮肤统计、阶段变化、动作确认、超时、动作 / 对局上限、模板歧义和时效、并发锁、停止控制、Mac / Windows 模拟输入与真实权重回放。详见 [离线验收报告](desktop-v2-verification.json)。

当前未运行真实 Mac 或 Windows 游戏对局。这里的“通过”只指离线和模拟检查；Windows 安装与真实 DPI、Mac 屏幕截图、英雄确认及找局界面仍需在目标机器验收。定向战吼、磁力、复杂技能和选一子选项等流程仍受公共动作适配器限制，不会当成普通出牌自动执行。

HsBgAI 参考版本为 `997910c3962490513626948241fdbdb0bd39b4cd`，其实现依赖 Windows BepInEx、net48 和游戏程序集。新版独立实现状态机，不包含该项目的 C# 源码、DLL 或统计数据，也不要求安装它的插件。
