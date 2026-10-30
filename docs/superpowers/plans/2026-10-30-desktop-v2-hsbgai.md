# 桌面 V2 实施计划

**目标：** 参考 HsBgAI，为 Mac 和 Windows 交付独立的状态机脚本、训练模型决策和可校准的对局衔接。

**架构：** 新建 `desktop_tavern` 包。公共日志与模型适配器继续使用 `live_tavern`；新增阶段识别、英雄统计、截图场景确认、动作确认控制器和会话记录。两平台使用各自的窗口执行器及独立配置。

**技术：** Python 3.10+、PyTorch、NumPy、Pillow，Windows 使用 PyAutoGUI，Mac 使用 PyObjC。

## 状态和数据

- 新建 `desktop_tavern/state.py`：扩展公开日志解析，保留 choice type，区分英雄、发现、招募、战斗和完成。英雄按物理槽位，检查本地归属和锁定状态。
- 新建 `desktop_tavern/heroes.py`：校验本地统计，支持 cardId / DBF 匹配，明确排序与回退依据。
- 新建 `desktop_tavern/session.py`：跨平台 OS 文件锁、逐局事件及局面快照；运行输出目录排除出 Git。
- 使用真实格式的合成日志验证阶段和英雄；真实格式证据来自本地 Tavern Ledger 的匿名样例，只引用相关结构，不分发完整原始对局。

## 场景和控制器

- 新建 `desktop_tavern/vision.py`：用户采集的局部模板、窗口截图映射、归一化误差、唯一匹配要求，模板证据每次点击前重新读取。
- 新建 `desktop_tavern/controller.py`：模型决定游戏内动作，统计决定英雄。维护单次待确认动作、超时、动作和对局上限；大厅与结算需明确校准和启用。
- 新建 `desktop_tavern/__main__.py`：初始化配置、检查、点位 / 行 / 场景校准、回放、运行、停止和重置。默认推荐模式。
- 对动作前状态变化、英雄确认、场景歧义、重复点击与超时使用模拟时钟 / 输入验收，确保没有真实鼠标事件。

## 双平台入口和交付

- 新建 `desktop_tavern/config.mac.example.json`、`config.windows.example.json`、`requirements.txt` 和 `.command` / `.cmd` 入口。
- 新建 `docs/DESKTOP_V2.md`，写出完整安装路径、权限、模型、统计、截图模板、执行参数和当前支持范围。
- 新建 `tools/verify_desktop_v2.py` 与 `tools/package_desktop_v2.py`。复用模型样例，分别输出 Mac / Windows ZIP，排除本地配置、真实截图和日志。
- 在新 venv 或已有已验证 venv 中运行检查；两份 ZIP 解压后完成模型回放及 CLI 检查。保存明确的离线报告。
- 提交相关文件，安全合入用户主工作树，保留已有未提交修改；推送到 GitHub 并验证远程提交。
