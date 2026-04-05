# Gemma 4

Gemma 4 是来自 Google 的多模态模型系列，支持文本、图像、音频和思考（链式推理）。这个统一模块处理所有 Gemma 4 变体——从轻量级 2B 到密集 31B 和 MoE 26B。

能力：
- **文本生成** — 单轮和多轮对话
- **思考模式** — 链式推理，使用 `--enable-thinking`
- **图像理解** — 单图像和多图像分析
- **音频理解** — 语音转录和音频分析（2B/4B 模型）
- **混合专家** — 使用 SwitchGLU 和 gather_mm 的稀疏 MoE（26B-A4B）

## 模型

| 模型 | 类型 | 参数 | 内存 | 视觉 | 音频 | K-eq-V |
|-------|------|--------|--------|-------|--------|
| `google/gemma-4-e2b-it` | Dense | 2B | ~5 GB | 是 | 是 | 否 |
| `google/gemma-4-e4b-it` | Dense | 4B | ~16 GB | 是 | 是 | 否 |
| `google/gemma-4-26b-a4b-it` | MoE | 26B (4B active) | ~52 GB | 是 | 否 | 是 |
| `google/gemma-4-31b-it` | Dense | 31B | ~63 GB | 是 | 否 | 是 |

> **K-eq-V**：全注意力层重用键投影作为值（没有单独的 `v_proj`），减少参数和内存，同时使用 `num_global_key_value_heads` 作为 KV 维度。

## 安装

```sh
pip install -U mlx-vlm
```

## CLI

### 文本生成

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "What is the capital of France?" \
  --max-tokens 500 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### 图像理解

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 500 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### 音频理解（仅 2B/4B）

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --audio path/to/audio.wav \
  --prompt "Transcribe this audio" \
  --max-tokens 500 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### 思考模式（链式推理）

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "I want to do a car wash that is 50 meters away, should I walk or drive?" \
  --enable-thinking \
  --max-tokens 2000 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### 图像 + 思考

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-31b-it \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --enable-thinking \
  --max-tokens 2000 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

### 带预算的思考

```sh
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "Explain quantum entanglement." \
  --enable-thinking \
  --thinking-budget 512 \
  --max-tokens 2000 \
  --temperature 1.0 --top-p 0.95 --top-k 64
```

## Python

### 基本文本生成

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config, "Write a poem about the ocean."
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

### 图像理解

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-31b-it")

image = ["path/to/image.jpg"]
prompt = apply_chat_template(
    processor, model.config, "Describe this image.",
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=image,
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

### 音频理解

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config, "Transcribe this audio",
    num_audios=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    audio=["path/to/audio.wav"],
    max_tokens=500,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

### 思考模式

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-e4b-it")

prompt = apply_chat_template(
    processor, model.config,
    "Explain quantum entanglement in simple terms.",
    chat_template_kwargs={"enable_thinking": True},
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    max_tokens=2000,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
print(result)
```

## 架构

所有 Gemma 4 变体共享一个具有条件功能的通用架构：

- **滑动 + 全注意力** — 5 个滑动窗口层 + 1 个全注意力层的模式
- **KV 共享** — 后层重用前层的 K/V（2B/4B 模型）
- **K-eq-V 注意力** — 全注意力层使用键状态作为值（26B/31B 模型）
- **逐层输入** — 额外的逐层 token 嵌入（2B/4B 模型）
- **MoE** — 128 个专家，使用 `gather_mm` 的 top-8 路由（仅 26B 模型）
- **SigLIP2 视觉编码器** — 在所有变体之间共享
- **Conformer 音频编码器** — 12 个 conformer 块（仅 2B/4B 模型）

## 说明

- 思考模式使用 `<|channel>...<channel|>` 分隔符将推理与最终答案分开。
- 音频使用基于 Conformer 的编码器，输入为 128-bin mel 频谱图（16kHz）。Audio token 根据持续时间动态扩展（~40ms/token，最多 750 个 token）。
- 支持的音频格式：WAV、MP3、FLAC 以及由 `soundfile` 处理的其他格式。
- 26B MoE 模型通过 `mlx_lm.SwitchGLU` 使用稀疏专家路由，使用 `mx.gather_mm` —— 每个token 只计算 top-8 专家。
- 31B 密集模型在 300 GB/s 系统上带宽受限，速度约为 ~5 tok/s（62.5 GB / 300 GB/s）。
