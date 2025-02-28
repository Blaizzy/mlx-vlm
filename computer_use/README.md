# Computer Use with MLX-VLM

<div align="center">

![MLX-VLM Computer Control](https://img.shields.io/badge/MLX--VLM-Computer%20Control-blue)
![macOS](https://img.shields.io/badge/platform-macOS-lightgrey)
![Apple Silicon](https://img.shields.io/badge/optimized-Apple%20Silicon-orange)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

A powerful tool that leverages Vision Language Models (VLMs) to enable AI-driven control of your Mac through visual understanding and contextual reasoning.

<p align="center">
  <i>Automate your workflow with natural language commands and visual intelligence</i>
</p>

## 🤖 Current Implementation Status
The current implementation is a GUI Agent (Level 1) with basic visual understanding and action capabilities. We're developing Level 2 (Autonomous GUI Agent) with enhanced memory, planning, reasoning, and human-in-the-loop functionality.
*Community help is more than welcome!* We're looking for contributors to help us reach Level 2 capabilities faster. Join us in building the future of computer automation.


## 🔍 Overview

Computer Use with MLX-VLM transforms how you interact with your Mac by combining the power of:

- **MLX** - Apple's machine learning framework optimized for Apple Silicon
- **Vision Language Models (VLMs)** - AI models that understand both visual and textual information
- **Automation** - Seamless execution of tasks across your Mac's interface

By processing screenshots and visual information from your screen, the system understands the current state of applications and executes appropriate actions to accomplish tasks you specify in natural language.

## ✨ Key Features

- **Mac-Native Performance**: Optimized for Apple Silicon with MLX for efficient, local processing
- **Visual Understanding**: Interprets screen content, UI elements, and application states
- **Contextual Reasoning**: Makes intelligent decisions based on visual context
- **Cross-Application Automation**: Works across multiple applications and system interfaces
- **Natural Language Control**: Simple, human-like instructions to control your computer
- **Privacy-Focused**: All processing happens locally on your device
- **Customizable**: Adapt to your specific workflow and preferences

## 🚀 Getting Started

### Prerequisites

- **macOS** running on Apple Silicon (M series)
- **Python 3.8+**
- **pip** (Python package manager)

### Installation
1. **Install MLX-VLM package**:
   ```bash
   pip install mlx-vlm
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Blaizzy/mlx-vlm.git
   ```

3. **Navigate to computer control directory**:
   ```bash
   cd computer_use
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Quick Start

Launch the application with:

```bash
python main.py
```

### Voice Control Interface

For hands-free operation, you can use the voice-enabled interface:

```bash
python main_voice.py
```

This launches a voice-controlled version of Computer Use that:
- Listens for your voice commands using your Mac's microphone
- Converts speech to text using local speech recognition using [mlx-whisper](https://github.com/ml-explore/mlx-examples).
- Processes your commands and executes them visually
- Provides voice feedback on actions taken (comming soon)

Voice commands work just like text commands, so you can say things like:

### Command Examples

Control your Mac with natural language instructions like:

```
"Open Safari and navigate to apple.com"
"Find the document I was working on yesterday and open it"
"Take a screenshot of this window and save it to my Desktop"
"Organize my Downloads folder by file type"
"Reply to the most recent email from my boss"
```

## ⚙️ How It Works

1. **Screen Capture**: The system takes screenshots of your Mac display
2. **Visual Analysis**: MLX-VLM processes the visual information to understand:
   - UI elements and their states
   - Text content on screen
   - Application context
   - System status
3. **Instruction Processing**: Your natural language commands are interpreted
4. **Action Planning**: The system determines the sequence of actions needed (comming soon)
5. **Execution**: Actions are performed through macOS APIs or simulated inputs (click, scroll, etc)

## 🔒 Privacy & Security

- **Local Processing**: All AI inference happens on your Mac using MLX
- **No Cloud Dependency**: Your screenshots and data never leave your device
- **Permission Control**: Fine-grained control over what the system can access
- **Transparent Operation**: Clear visibility into actions being performed

## 🛠️ Troubleshooting

### Common Issues

- **Permission Errors**: Make sure to grant screen recording permissions in System Preferences > Security & Privacy > Privacy
- **Performance Issues**: Try reducing the screenshot resolution in config.json
- **Application Compatibility**: Some applications with non-standard UI elements may have limited support

### Getting Help

- Check the [Issues](https://github.com/yourusername/computer_use/issues) page
- Join our [Discord community](https://discord.gg/yourdiscord)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create your feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**:
   ```bash
   python -m pytest
   ```
5. **Commit your changes**:
   ```bash
   git commit -m 'Add some amazing feature'
   ```
6. **Push to the branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The **MLX team at Apple** for creating the MLX framework
- **Our community of testers and contributors** who help improve the project

---

<p align="center">
  <i>Made with ❤️ for Mac users who love automation and AI</i>
</p>