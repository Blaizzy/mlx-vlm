# 使用 MLX-VLM 进行计算机控制

<div align="center">

![MLX-VLM Computer Control](https://img.shields.io/badge/MLX--VLM-Computer%20Control-blue)
![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)
![Apple Silicon](https://img.shields.io/badge/optimized-Apple%20Silicon-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

一个强大的工具，利用视觉语言模型（VLM）通过视觉理解和上下文推理实现 AI 驱动的 Mac 控制。

<p align="center">
  <i>使用自然语言命令和视觉智能自动化您的工作流程</i>
</p>

## 🤖 当前实现状态
项目现在支持 Level 1（GUI 代理）和 Level 2（自主 GUI 代理）功能：

- **Level 1（GUI 代理）**：基本的视觉理解和操作能力
- **Level 2（自主 GUI 代理）**：增强的记忆、规划、推理和人工在环功能

*社区帮助非常受欢迎！* 我们正在寻找贡献者来帮助我们进一步增强这些功能。加入我们，共同构建计算机自动化的未来。

## 🔍 概述

使用 MLX-VLM 进行计算机控制改变了您与 Mac 的交互方式，结合了以下强大功能：

- **MLX** - Apple 为 Apple Silicon 优化的机器学习框架
- **视觉语言模型（VLM）** - 理解视觉和文本信息的 AI 模型
- **自动化** - 在 Mac 界面上无缝执行任务

通过处理来自屏幕的截图和视觉信息，系统能够理解应用程序的当前状态，并执行适当的操作来完成您以自然语言指定的任务。

## ✨ 主要功能

- **Mac 原生性能**：使用 MLX 为 Apple Silicon 优化，实现高效的本机处理
- **视觉理解**：解读屏幕内容、UI 元素和应用程序状态
- **上下文推理**：基于视觉上下文做出智能决策
- **跨应用程序自动化**：在多个应用程序和系统界面中工作
- **自然语言控制**：使用简单、类似人类的指令控制您的计算机
- **隐私保护**：所有处理都在您的设备上本地进行
- **可定制**：适应您的特定工作流程和偏好
- **自主操作**：Level 2 代理可以规划和执行多步骤任务，只需最小监督
- **语音控制**：使用本地语音识别进行免提操作

## 🚀 快速开始

### 先决条件

- 运行在 Apple Silicon（M 系列）上的 **macOS**
- **Python 3.8+**
- **pip**（Python 包管理器）

### 安装
1. **安装 MLX-VLM 包**：
   ```bash
   pip install mlx-vlm
   ```

2. **克隆仓库**：
   ```bash
   git clone https://github.com/Blaizzy/mlx-vlm.git
   ```

3. **导航到计算机控制目录**：
   ```bash
   cd computer_use
   ```

4. **安装依赖项**：
   ```bash
   pip install -r requirements.txt
   ```

## 💻 使用方法

### 快速开始

使用以下命令启动标准应用程序：

```bash
python main.py
```

### 自主 GUI 代理

对于具有规划功能的增强自主操作：

```bash
python autonomous_gui_agent.py
```

这将启动 Level 2 自主代理，它可以：
- 规划并执行多步骤任务
- 在操作之间维护上下文
- 基于视觉反馈做出决策
- 在需要时请求人工协助

### 语音控制界面

对于免提操作，您可以使用支持语音的自主代理：

```bash
python autonomous_gui_agent_voice.py
```

这将启动语音控制版本，它可以：
- 使用 Mac 麦克风监听您的语音命令
- 使用本地语音识别（通过 [mlx-whisper](https://github.com/ml-explore/mlx-examples)）将语音转换为文本
- 处理您的命令并视觉化执行
- 对所采取的操作提供音频反馈

语音命令的工作方式与文本命令相同，因此您可以这样说：

### 命令示例

使用自然语言指令控制您的 Mac，例如：

```
"打开 Safari 并导航到 apple.com"
"打开通知选项卡并点击第一个通知"
"打开邮件应用程序并回复最新的电子邮件"
```

## ⚙️ 工作原理

1. **屏幕捕获**：系统截取 Mac 显示器的截图
2. **视觉分析**：MLX-VLM 处理视觉信息以理解：
   - UI 元素及其状态
   - 屏幕上的文本内容
   - 应用程序上下文
   - 系统状态
3. **指令处理**：您的自然语言命令被解释
4. **操作规划**：系统确定所需操作序列
5. **执行**：通过 macOS API 或模拟输入（（点击、滚动等）执行操作

## 🔒 隐私与安全

- **本地处理**：所有 AI 推理都在您的 Mac 上使用 MLX 进行
- **无云依赖**：您的截图和数据永远不会离开您的设备
- **权限控制**：精确控制系统可以访问的内容
- **透明操作**：清晰了解正在执行的操作

## 🛠️ 故障排除

### 常见问题

- **权限错误**：确保在系统偏好设置 > 安全性与隐私 > 隐私中授予屏幕录制权限
- **性能问题**：尝试在 config.json 中降低截图分辨率
- **应用程序兼容性**：一些具有非标准 UI 元素的应用程序可能支持有限

### 获取帮助

- 查看 [Issues](https://github.com/yourusername/computer_use/issues) 页面
- 加入我们的 [Discord 社区](https://discord.gg/yourdiscord)

## 🤝 贡献

我们欢迎贡献！以下是入门方法：

1. **Fork 仓库**
2. **创建您的功能分支**：
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **进行更改**
4. **运行测试**：
   ```bash
   python -m pytest
   ```
5. **提交您的更改**：
   ```bash
   git commit -m 'Add some amazing feature'
   ```
6. **推送到分支**：
   ```bash
   git push origin feature/amazing-feature
   ```
7. **打开 Pull Request**

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细指南。

## 📜 许可证

本项目在 MIT 许可证下发布 - 有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **Apple 的 MLX 团队**创建了 MLX 框架
- **我们的测试者和贡献者社区**帮助改进该项目

---

<p align="center">
  <i>为热爱自动化和 AI 的 Mac 用户而制作，用 ❤️ 打造</i>
</p>
