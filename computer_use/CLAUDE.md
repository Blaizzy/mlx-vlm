# Computer Use - 计算机控制模块

[根目录](../CLAUDE.md) > **computer_use**

## 模块职责

基于视觉语言模型（VLM）的计算机 GUI 自动化系统，让 AI 能够"看"到屏幕内容并执行操作，实现自然语言控制 Mac 的能力。

## 入口与启动

### 主要脚本

#### Level 1: GUI Agent（基础版本）
```bash
python gui_agent.py
```
功能：
- 实时屏幕捕获
- 视觉理解和 UI 分析
- 鼠标点击、键盘输入等基础操作
- 单步指令执行

#### Level 2: Autonomous GUI Agent（增强版本）
```bash
python autonomous_gui_agent.py
```
功能：
- 多步任务规划和执行
- 上下文记忆管理
- 错误恢复和重试
- 人在回路（Human-in-the-loop）支持

#### 语音控制版本
```bash
# 基础语音控制
python gui_agent_voice.py

# 自主语音控制
python autonomous_gui_agent_voice.py
```
功能：
- 本地语音识别（mlx-whisper）
- 语音指令解析
- 音频反馈

## 对外接口

### GUI Agent API

#### `GUIAgent` 类（`gui_agent.py`）
```python
class GUIAgent:
    def __init__(self, model_path: str, ...):
        ...

    def capture_screen(self) -> Image:
        """捕获当前屏幕"""

    def understand_scene(self, screenshot: Image) -> str:
        """理解屏幕内容和 UI 元素"""

    def execute_action(self, action: str) -> bool:
        """执行 GUI 操作（点击、输入等）"""
```

#### `AutonomousGUIAgent` 类（`autonomous_gui_agent.py`）
```python
class AutonomousGUIAgent:
    def __init__(self, model_path: str, ...):
        ...

    def plan_task(self, instruction: str) -> List[Action]:
        """将复杂指令分解为操作序列"""

    def execute_plan(self, plan: List[Action]) -> bool:
        """执行操作计划"""

    def update_context(self, result: ActionResult):
        """更新执行上下文和记忆"""
```

### 使用示例

#### Python API
```python
from computer_use import GUIAgent

# 初始化
agent = GUIAgent(model_path="mlx-community/Qwen2-VL-2B-Instruct-4bit")

# 执行指令
instruction = "Open Safari and navigate to apple.com"
success = agent.execute_instruction(instruction)

print(f"Success: {success}")
```

#### 命令行
```bash
# 启动交互式 agent
python gui_agent.py --model mlx-community/Qwen2-VL-2B-Instruct-4bit

# 执行单条指令
python gui_agent.py --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --instruction "Open the notifications tab"
```

## 关键依赖与配置

### 依赖项

```
# computer_use/requirements.txt
mlx-vlm
pyautogui  # GUI 自动化
Pillow  # 图像处理
pyobjc  # macOS API
mlx-whisper  # 本地语音识别（语音版本）
```

### 配置文件

#### `config.json`
```json
{
  "model_path": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
  "screenshot_interval": 1.0,
  "max_steps": 10,
  "confidence_threshold": 0.7,
  "voice_enabled": false
}
```

### 系统权限

- **屏幕录制**: 系统偏好设置 > 隐私与安全性 > 屏幕录制
- **辅助功能**: 系统偏好设置 > 隐私与安全性 > 辅助功能
- **麦克风**: 语音控制版本需要

## 数据模型

### 执行流程

```
用户指令（自然语言）
    ↓
指令解析（VLM 理解）
    ↓
任务规划（Level 2）
    ↓
屏幕捕获
    ↓
视觉理解（VLM 分析）
    ↓
动作生成（点击/输入/滚动）
    ↓
执行操作（PyAutoGUI/pyobjc）
    ↓
结果验证
    ↓
更新上下文（Level 2）
    ↓
完成/重试/请求帮助
```

### 关键数据结构

#### `Action`（操作）
```python
@dataclass
class Action:
    type: str  # "click", "type", "scroll", "wait"
    position: Tuple[int, int]  # (x, y) for click
    text: str  # text for type
    duration: float  # duration for wait
```

#### `ActionResult`（操作结果）
```python
@dataclass
class ActionResult:
    success: bool
    screenshot: Image
    error: Optional[str]
    observation: str
```

#### `Context`（上下文，Level 2）
```python
@dataclass
class Context:
    history: List[Action]
    current_screen: Image
    task_goal: str
    step_number: int
    memory: Dict[str, Any]
```

## 测试与质量

### 测试方法

#### 手动测试
```bash
# 运行基础 agent
python gui_agent.py

# 运行自主 agent
python autonomous_gui_agent.py

# 测试语音控制
python autonomous_gui_agent_voice.py
```

#### 测试场景
- 基础操作：打开应用、点击按钮、输入文本
- 复杂任务：多步骤工作流、跨应用操作
- 错误处理：UI 元素未找到、操作失败恢复
- 语音控制：语音识别准确率、噪音环境

### 质量指标

- **成功率**: 任务完成率 > 80%
- **响应时间**: 单步操作 < 5 秒
- **准确率**: 视觉理解准确率 > 90%
- **稳定性**: 连续运行 1 小时无崩溃

## 常见问题 (FAQ)

### Q: 如何调试屏幕捕获问题？

A: 检查系统权限和屏幕分辨率：
```python
import pyautogui
print(pyautogui.size())  # 屏幕尺寸
screenshot = pyautogui.screenshot()
screenshot.save("debug.png")  # 保存截图检查
```

### Q: 语音识别不准确怎么办？

A: 调整麦克风设置或使用文本输入：
1. 检查系统麦克风权限
2. 在安静环境中使用
3. 使用文本指令作为备选
4. 调整 mlx-whisper 模型（更大模型 = 更高准确率）

### Q: 如何添加新的 GUI 操作？

A: 在 `gui_agent.py` 中扩展：
```python
def execute_action(self, action: Action) -> bool:
    if action.type == "drag":
        # 实现拖拽操作
        pyautogui.dragTo(action.position, duration=action.duration)
    elif action.type == "hotkey":
        # 实现快捷键
        pyautogui.hotkey(*action.keys.split('+'))
    # ... 其他操作
```

### Q: 如何限制 agent 的操作范围？

A: 在配置中设置白名单：
```json
{
  "allowed_apps": ["Safari", "Finder", "Notes"],
  "forbidden_actions": ["delete", "format"]
}
```

### Q: Level 1 和 Level 2 agent 如何选择？

A:
- **Level 1 (GUI Agent)**: 简单单步任务，快速响应
- **Level 2 (Autonomous Agent)**: 复杂多步任务，需要规划和记忆

## 相关文件清单

### 核心脚本
- `gui_agent.py`: Level 1 GUI Agent
- `autonomous_gui_agent.py`: Level 2 Autonomous GUI Agent
- `gui_agent_voice.py`: 语音控制 GUI Agent
- `autonomous_gui_agent_voice.py`: 语音控制 Autonomous Agent
- `utils.py`: 工具函数（屏幕捕获、操作执行）

### 数据和资源
- `audio/`: 音频文件（语音反馈）
- `screenshots/`: 截图示例
- `navigation_history.csv`: 导航历史记录
- `requirements.txt`: Python 依赖

### 文档
- `README.md`: 详细使用指南

## 变更记录 (Changelog)

### 2026-04-14 - 初始化模块文档
- ✅ **创建模块文档**: 完成 Computer Use 模块 CLAUDE.md
- 🏗️ **架构说明**: Level 1/2 功能、执行流程、API 文档
- 🔧 **使用指南**: 安装、配置、常见问题

### 功能演进
- **v1.0**: 基础 GUI Agent（单步操作）
- **v2.0**: Autonomous GUI Agent（规划和记忆）
- **v2.1**: 语音控制支持
- **未来**: 多显示器支持、更复杂的任务规划

---

**维护者**: MLX-VLM Team | **测试覆盖**: 手动测试 | **文档状态**: 完善
