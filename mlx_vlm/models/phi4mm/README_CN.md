# Phi-4 多模态 (phi4mm)

Phi-4 多模态是一个三模态模型，支持文本、图像和音频理解。

## 架构

| 组件 | 详情 |
|---|---|
| **语言模型** | Phi-4 (32 层, 3072 隐藏, 24 头, 8 KV 头) |
| **视觉编码器** | SigLIP-2 (27 层, 1152 隐藏, 16 头) |
| **音频编码器** | 级联 Conformer (24 块, 1024 维, 16 头) |
| **视觉投影器** | 2 层 MLP (1152 → 3072 → 3072, GELU) |
| **音频投影器** | 带有语音/视觉模式的 2 层 MLP |

### LoRA 切换

原始检查点带有应用于 LLM 骨干的 **两个 LoRA 适配器**：

- **视觉 LoRA** (r=256, alpha=512) — 默认在加载时合并。
- **语音 LoRA** (r=320, alpha=640) — 存储用于运行时切换。

`set_modality()` 根据输入类型自动选择正确的 LoRA（或两者）。

## 模型

- Hugging Face ID: `microsoft/Phi-4-multimodal-instruct`
- 远程处理器代码已在树内移植，因此 `--trust-remote-code` 是可选的。

## CLI

### 文本理解

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --prompt "Explain theory of relativity in simple terms." \
  --max-tokens 256
```

### 图像理解

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

### 音频理解

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --audio /path/to/audio.wav \
  --prompt "" \
  --max-tokens 256
```

### 多模态（图像 + 音频）

```sh
mlx_vlm.generate \
  --model microsoft/Phi-4-multimodal-instruct \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "" \
  --max-tokens 256
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("microsoft/Phi-4-multimodal-instruct")

image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = "What animals are in image?"

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
    num_audios=len(audio),
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=image,
    audio=audio,
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
```

## 量化

```sh
mlx_vlm.convert \
  --model microsoft/Phi-4-multimodal-instruct \
  -q \
  --mlx-path Phi-4-multimodal-instruct-4bit
```

在量化期间，模型预先将两个 LoRA 适配器合并到 LLM 权重中
并且仅量化语言模型。视觉编码器、音频编码器和
投影器保留在 bfloat16 中。

量化后，LoRA 切换被禁用（不需要，因为两个适配器
都已烘焙在内）。

## 注意事项

- 音频输入是 16 kHz 单声道波形；处理器自动处理重采样。
- `<|image_1|>` / `<|audio_1|>` 占位符由 `apply_chat_template` 插入。
