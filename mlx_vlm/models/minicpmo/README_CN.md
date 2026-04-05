# MiniCPM-o (MiniCPM-o-4_5)

MiniCPM-o 是一个支持文本、图像和音频理解的 omni 模型。

此 MLX-VLM 集成包括：
- 自定义 MiniCPM-o 处理器注册
- 图像 + 音频输入预处理
- `<image>` 和 `<audio>` 的提示词占位符处理

## 模型

- Hugging Face ID：`openbmb/MiniCPM-o-4_5`
- 远程 tokenizer/处理器代码在树内移植，因此不需要 `--trust-remote-code`。

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 图像理解

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --image /path/to/image.jpg \
  --prompt "Describe this image briefly." \
  --max-tokens 128 \
  --temperature 0
```

### 音频理解

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --audio /path/to/audio.wav \
  --prompt "Describe this audio briefly." \
  --max-tokens 256 \
  --temperature 0
```

### 多模态（图像 + 音频）

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "Describe what you see and hear." \
  --max-tokens 256 \
  --temperature 0
```

### 禁用聊天模板中的思考

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --audio /path/to/audio.wav \
  --prompt "Describe this audio briefly." \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --max-tokens 256 \
  --temperature 0
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(
    "openbmb/MiniCPM-o-4_5",
)

image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = "Summarize the visual and audio content."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
    num_audios=len(audio),
    chat_template_kwargs={"enable_thinking": False},
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

## 说明

- 使用 `apply_chat_template` 时，通常不应手动添加 `<image>` 或 `<audio>` 栩记。
- 对于 CLI，`--audio` 目前在模板计数中的行为为单音频路径。
