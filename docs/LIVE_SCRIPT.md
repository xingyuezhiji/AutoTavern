# 用训练模型驱动酒馆战棋客户端

`live_tavern` 是 Windows 客户端试运行脚本：读取 Power.log，加载你训练的 PyTorch 权重，输出下一步建议，也能通过校准后的鼠标坐标执行支持的动作。默认输出建议。Mac 可以做日志回放和模型推理。

当前交付是实验性 pilot，尚未在正式 Windows 客户端实测。它不是已经验收的完整天梯机器人。日志能还原金币、星级、手牌、商店和场面，但部分效果计数、饰品状态和对手历史仍缺失。记录中的 `coverage=estimated_log_context` 与 `unobserved` 会列出这些限制；不能把这个输入当成精确的模拟器状态。

## 先在 Windows 安装

需要 Python 3.10 或更新版本。将整个包解压到 Windows；在源码仓库中也可直接运行：

```bat
live_tavern\setup-windows.cmd
```

安装后编辑 `live_tavern\config.local.json`：

- `player_name`：日志中实际显示的本地玩家名，要求完全匹配，通常包含 `#` 后缀。
- `log_root`：实际炉石安装目录下的 `Logs`，也可直接填完整 `Power.log` 路径。不同机器的目录可能不同，以实际文件为准。
- `window_title`：客户端窗口的完整标题。
- `accepted_builds`：核验卡牌与日志格式后接受的客户端 BuildNumber，例如 `["253932"]`。默认空列表，执行模式会等待。当前训练卡库来自客户端 `36.6.3.253932`，新补丁不能仅靠修改这项就视为兼容。

在 `%LocalAppData%\Blizzard\Hearthstone\log.config` 中启用日志，修改后重启客户端：

```ini
[Power]
LogLevel=1
FilePrinting=true
ConsolePrinting=false
ScreenPrinting=false
Verbose=true
```

脚本依赖 `DebugPrintOptions` 的合法选项。没有这类行时会等待，不会自行猜合法动作。日志中的攻击、血量、金币或物理位置缺失时同样等待。已购随从的打出费用为零；零金币不会按回合数补成其他数值。

## 加载权重检查

把自己的 checkpoint 放到本地路径。便携包包含已验证的 917730 权重，位于 `models\tavern_bc_policy.pt`。

```bat
.venv-live\Scripts\python.exe -m live_tavern --config live_tavern\config.local.json --checkpoint models\tavern_bc_policy.pt doctor
```

使用训练代码的 `ModelConfig` 与 `build_model_class`，支持 MLP、动作身份头及实体 Transformer；不会重新训练，也不上传权重。CPU 推理即可运行。输入严格要求 4017 个 floats、447 个卡牌编号、245 个动作；checkpoint 的目录指纹必须与当前训练卡库一致。

## 校准鼠标位置

示例配置没有预填坐标。正式执行只使用你记录的位置。保持相同客户端尺寸；窗口移动时会重新映射屏幕原点，执行手势过程中移动窗口则停止。

例如校准购买后的手牌落点：

```bat
.venv-live\Scripts\python.exe -m live_tavern calibrate --point hand_drop
```

命令启动后，把炉石切到前台，鼠标移到对应落点，按 **F6** 记录。按 **F8** 退出。

校准当前有三件商品的商店，从左到右分别记录中心：

```bat
.venv-live\Scripts\python.exe -m live_tavern calibrate --row shop --count 3
```

每种数量都独立记录：商店 `shop`、手牌 `hand`、我方场面 `board`、发现选项 `choice`。手牌有弧形排列，直接记录实际卡牌中心，不用均匀直线代替。

`board_gaps` 记录插入位置。例如场上两个随从有三个插入位置：

```bat
.venv-live\Scripts\python.exe -m live_tavern calibrate --row board_gaps --count 3
```

可记录的按钮或落点：`end`、`refresh`、`freeze`、`upgrade`、`power0`、`power1`、`dark_gift`、`neutral`、`hand_drop`、`sell_drop`、`spell_drop`。未校准的动作不会进入可执行动作集合。只校准部分位置，就只支持对应的动作与槽位数量。

## 先看建议，再试执行

手动进入单人酒馆战棋、完成英雄选择后运行：

```bat
live_tavern\start-windows.cmd "C:\模型目录\tavern_bc_policy.pt"
```

输出包含动作编号、回合、剩余金币、前五个可执行动作的概率、具体鼠标计划和缺失上下文。记录保存在 `runs\live\decisions.jsonl`。

校准与版本核验完成后，开启实验性执行：

```bat
live_tavern\start-windows.cmd "C:\模型目录\tavern_bc_policy.pt" --execute --allow-estimated-context
```

`--allow-estimated-context` 表示接受日志编码仍有缺失，不能让核心状态缺失的动作通过。一次只执行一个模型动作；等待客户端动作确认和新状态后才继续。8 秒未确认、窗口失焦、坐标尺寸变化或状态过期时停止，不重复点击。每回合最多执行 60 个动作；模型选择结束回合后等待下一回合。

按 **F8**、**Ctrl+C** 或运行 `live_tavern\stop-windows.cmd` 停止。停止文件会保留；重启前删除 `live_tavern\STOP`。PyAutoGUI 的屏幕角落停止机制也保留。

## 当前支持范围

当前自动动作集合包含：客户端明确允许的购买、出售、刷新、冻结、升级、无目标英雄技能、无目标出牌、一般发现，以及日志明确给出的目标选择和移动操作。升级与刷新费用未明确记录时禁用对应动作。

全部 245 个动作有编号和鼠标计划映射，但**有映射不等于已完成客户端接入**：磁力、激活、黑暗之赐、带目标战吼/法术/英雄技能、Choose One 等流程，目前不会主动进入自动动作集合。它们涉及额外 UI 或目标阶段，仍需真实客户端日志与截图验收。脚本也不自动匹配、不负责选择英雄、不处理双人模式或重启客户端。

遇到未知新卡时停止该状态的决策，不把它映射为空卡。模型只在支持且校准的合法动作里选取，因此输出概率属于这个子集，不能当成完整动作空间上的概率。

## 离线回放

历史日志不会发送鼠标输入：

```bat
.venv-live\Scripts\python.exe -m live_tavern --checkpoint models\tavern_bc_policy.pt replay "C:\对局备份\Power.log"
```

便携包附带合成样例，可直接验证链路：

```bat
.venv-live\Scripts\python.exe -m live_tavern --config samples\geometry.json --checkpoint models\tavern_bc_policy.pt replay samples\Power.log
```

样例坐标仅用于离线回放，不得用于正式客户端。它不是采集到的真人对局。

源码安装了匹配版本的 `taverngym` 后，也可从模拟器 `state_json()` 的保存结果做精确推理：

```bat
.venv-live\Scripts\python.exe -m live_tavern --checkpoint models\tavern_bc_policy.pt sim-replay saved-state.json --seat 0
```

便携包不包含 Rust 模拟器，`sim-replay` 在该包中需额外安装匹配的 TavernGym；普通日志模式不需要 Rust。

## 已做的验证

使用 917730 的 2,992,951 参数权重，验证了完整模型加载、日志到购买手势、零金币、未知卡牌拒绝、位置缺失拒绝、过期决策、目标实体与移动插入位置、245 个动作编号，以及日志截断和跨行读取。另用原生 TavernGym 对比精确 v2 编码和目录指纹，执行了 24 次合法模型决策。

结果在 `docs/live-script-verification.json`。日志动作验收使用合成的真实格式样例，模型步骤使用模拟器；两者都不证明正式客户端效果。

复跑命令：

```bat
.venv-live\Scripts\python.exe tools\verify_live_script.py --checkpoint models\tavern_bc_policy.pt --report runs\live\verification.json
```

## 参考项目与实际采用的思路

| 项目 | 本次参考内容 | 区别 |
| --- | --- | --- |
| [HSArriver-Turbo](https://github.com/Donald-Yip/HSArriver-Turbo) | 日志状态版本、客户端坐标映射、目标与手牌操作 | 主要面向构筑对战，不能直接套用酒馆动作 |
| [MaaTavernBot](https://github.com/0079527007/MaaTavernBot) | 酒馆 UI 的流程拆分 | 使用 ADB 模拟器，本脚本使用 Windows 前台客户端 |
| [Hearthstone-Script](https://github.com/xjw580/Hearthstone-Script) | 策略插件与运行控制 | 核心实现拆为 Kotlin 子模块，主要策略接口面向构筑 |
| [TavernOracle](https://github.com/adampan0527/TavernOracle/blob/main/README_CN.md) | HDT 实时公开状态、零金币、商品位置、每次操作后重新读状态 | 目前是手写策略的建议工具；本脚本使用本地训练权重 |

这些项目仅用于研究接口和设计，本次 Python 实现独立编写。TavernOracle 的准确率和延迟目标不作为本脚本效果承诺。
