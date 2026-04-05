[![Upload Python Package](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Blaizzy/mlx-vlm/actions/workflows/python-publish.yml)
# MLX-VLM

**[English](README.md)** | **[简体中文](README_CN.md)**

MLX-VLM 是一个在 Mac 上使用 MLX 进行视觉语言模型（VLMs）和 Omni 模型（支持音频和视频的 VLMs）推理与微调的软件包。

## 目录
- [安装](#安装)
- [使用](#使用)
  - [命令行界面（CLI）](#命令行界面cli)
    - [思考预算](#思考预算)
  - [Gradio 聊天界面](#gradio-聊天界面)
  - [Python 脚本](#python-脚本)
- [激活量化（CUDA）](#激活量化cuda)
- [多图像聊天支持](#多图像聊天支持)
  - [支持的模型](#支持的模型)
  - [使用示例](#使用示例)
- [模型特定文档](#模型特定文档)
- [视觉特征缓存](#视觉特征缓存)
- [TurboQuant KV 缓存](#turboquant-kv-缓存)
- [微调](#微调)

## 模型特定文档

某些模型有详细的文档，包括提示格式、示例和最佳实践：

| 模型 | 文档 |
|-------|---------------|
| DeepSeek-OCR | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/README.md) |
| DeepSeek-OCR-2 | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr_2/README.md) |
| DOTS-OCR | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md) |
| DOTS-MOCR | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/dots_ocr/README.md) |
| GLM-OCR | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/glm_ocr/README.md) |
| Phi-4 Reasoning Vision | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4_siglip/README.md) |
| MiniCPM-o | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minicpmo/README.md) |
| Phi-4 Multimodal | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi4mm/README.md) |
| MolmoPoint | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/README.md) |
| Moondream3 | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/moondream3/README.md) |
| Gemma 4 | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md) |
| Falcon-OCR | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/falcon_ocr/README.md) |
| Granite Vision 3.2 | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite_vision/README.md) |
| Granite 4.0 Vision | [文档](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/granite4_vision/README.md) |

## 安装

最简单的入门方法是使用 pip 安装 `mlx-vlm` 软件包：

```sh
pip install -U mlx-vlm
```

## 使用

### 命令行界面（CLI）

使用 CLI 从模型生成输出：

```sh
# 文本生成
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Hello, how are you?"

# 图像生成
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg

# 音频生成（新增）
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "Describe what you hear" --audio /path/to/audio.wav

# 多模态生成（图像 + 音频）
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "Describe what you see and hear" --image /path/to/image.jpg --audio /path/to/audio.wav
```

#### 思考预算

对于思考模型（例如 Qwen3.5），您可以限制在思考块中花费的令牌数量：

```sh
mlx_vlm.generate --model mlx-community/Qwen3.5-2B-4bit \
  --thinking-budget 50 \
  --thinking-start-token "<think>" \
  --thinking-end-token "</think>" \
  --enable-thinking \
  --prompt "Solve 2+2"
```

| 标志 | 描述 |
|------|-------------|
| `--enable-thinking` | 在聊天模板中激活思考模式 |
| `--thinking-budget` | 思考块内允许的最大令牌数 |
| `--thinking-start-token` | 打开思考块的令牌（默认：`<think>`） |
| `--thinking-end-token` | 关闭思考块的令牌（默认：`</think>`） |

当超过预算时，模型将被强制发出 `\n</think>` 并转换到答案。如果传递了 `--enable-thinking` 但模型的聊天模板不支持它，则只有当模型自己生成开始令牌时才会应用预算。

### Gradio 聊天界面

使用 Gradio 启动聊天界面：

```sh
mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Python 脚本

以下是在 Python 脚本中使用 MLX-VLM 的示例：

```python
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# 加载模型
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# 准备输入
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
# image = [Image.open("...")] 也可以与 PIL.Image.Image 对象一起使用
prompt = "Describe this image."

# 应用聊天模板
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# 生成输出
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

#### 音频示例

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# 加载支持音频的模型
model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

# 准备音频输入
audio = ["/path/to/audio1.wav", "/path/to/audio2.mp3"]
prompt = "Describe what you hear in these audio files."

# 应用带有音频的聊天模板
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_audios=len(audio)
)

# 生成带有音频的输出
output = generate(model, processor, formatted_prompt, audio=audio, verbose=False)
print(output)
```

#### 多模态示例（图像 + 音频）

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# 加载多模态模型
model_path = "mlx-community/gemma-3n-E2B-it-4bit"
model, processor = load(model_path)
config = model.config

# 准备输入
image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = ""

# 应用聊天模板
formatted_prompt = apply_chat_template(
    processor, config, prompt,
    num_images=len(image),
    num_audios=len(audio)
)

# 生成输出
output = generate(model, processor, formatted_prompt, image, audio=audio, verbose=False)
print(output)
```

### 服务器（FastAPI）

启动服务器：
```sh
mlx_vlm.server --port 8080

# 在启动时预加载模型（Hugging Face 仓库或本地路径）
mlx_vlm.server --model <hf_repo_or_local_path>

# 使用适配器预加载模型
mlx_vlm.server --model <hf_repo_or_local_path> --adapter-path <adapter_path>

# 启用信任远程代码（某些模型需要）
mlx_vlm.server --trust-remote-code
```

#### 服务器选项

- `--model`：在服务器启动时预加载模型，接受 Hugging Face 仓库 ID 或本地路径（可选，如果省略则在第一次请求时延迟加载）
- `--adapter-path`：用于预加载模型的适配器权重路径
- `--host`：主机地址（默认：`0.0.0.0`）
- `--port`：端口号（默认：`8080`）
- `--trust-remote-code`：从 Hugging Face Hub 加载模型时信任远程代码
- `--kv-bits`：KV 缓存量化的位数（例如 `3.5` 用于 TurboQuant）
- `--kv-quant-scheme`：KV 缓存量化后端（`uniform` 或 `turboquant`）

您也可以通过环境变量设置信任远程代码：
```sh
MLX_TRUST_REMOTE_CODE=true mlx_vlm.server
```

服务器为不同用例提供多个端点，并支持动态模型加载/卸载与缓存（一次一个模型）。

#### 可用端点

- `/models` 和 `/v1/models` - 列出本地可用的模型
- `/chat/completions` 和 `/v1/chat/completions` - OpenAI 兼容的聊天风格交互端点，支持图像、音频和文本
- `/responses` 和 `/v1/responses` - OpenAI 兼容的响应端点
- `/health` - 检查服务器状态
- `/unload` - 从内存中卸载当前模型

#### 使用示例

##### 列出可用模型

```sh
curl "http://localhost:8080/models"
```

##### 文本输入

```sh
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you"
      }
    ],
    "stream": true,
    "max_tokens": 100
  }'
```

##### 图像输入

```sh
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-VL-32B-Instruct-8bit",
    "messages":
    [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "This is today's chart for energy demand in California. Can you provide an analysis of the chart and comment on the implications for renewable energy in California?"
          },
          {
            "type": "input_image",
            "image_url": "/path/to/repo/examples/images/renewables_california.png"
          }
        ]
      }
    ],
    "stream": true,
    "max_tokens": 1000
  }'
```

##### 音频支持（新增）
```sh
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3n-E2B-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "text", "text": "Describe what you hear in these audio files" },
          { "type": "input_audio", "input_audio": "/path/to/audio1.wav" },
          { "type": "input_audio", "input_audio": "https://example.com/audio2.mp3" }
        ]
      }
    ],
    "stream": true,
    "max_tokens": 500
  }'
```

##### 多模态（图像 + 音频）
```sh
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3n-E2B-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "input_image", "image_url": "/path/to/image.jpg"},
          {"type": "input_audio", "input_audio": "/path/to/audio.wav"}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

##### 响应端点
```sh
curl -X POST "http://localhost:8080/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "What is in this image?"},
          {"type": "input_image", "image_url": "/path/to/image.jpg"}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

#### 请求参数

- `model`：模型标识符（必需）
- `messages`：聊天/ OpenAI 端点的聊天消息
- `max_tokens`：最大生成令牌数
- `temperature`：采样温度
- `top_p`：Top-p 采样参数
- `top_k`：Top-k 采样截断
- `min_p`：Min-p 采样阈值
- `repetition_penalty`：应用于重复令牌的惩罚
- `stream`：启用流式响应


## 激活量化（CUDA）

在使用 MLX CUDA 在 NVIDIA GPU 上运行时，使用 `mxfp8` 或 `nvfp4` 模式量化的模型需要激活量化才能正常工作。这会将 `QuantizedLinear` 层转换为 `QQLinear` 层，该层对权重和激活值都进行量化。

### 命令行

使用 `-qa` 或 `--quantize-activations` 标志：

```sh
mlx_vlm.generate --model /path/to/mxfp8-model --prompt "Describe this image" --image /path/to/image.jpg -qa
```

### Python API

将 `quantize_activations=True` 传递给 `load` 函数：

```python
from mlx_vlm import load, generate

# 启用激活量化加载
model, processor = load(
    "path/to/mxfp8-quantized-model",
    quantize_activations=True
)

# 正常生成
output = generate(model, processor, "Describe this image", image=["image.jpg"])
```

### 支持的量化模式

- `mxfp8` - 8 位 MX 浮点
- `nvfp4` - 4 位 NVIDIA 浮点

> **注意**：此功能是 CUDA 上 mxfp/nvfp 量化模型所必需的。在 Apple Silicon（Metal）上，这些模型不需要该标志即可工作。

## 多图像聊天支持

MLX-VLM 支持同时分析多个图像，适用于精选模型。此功能能够在单次对话中实现更复杂的视觉推理任务和跨多个图像的综合分析。


### 使用示例

#### Python 脚本

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = model.config

images = ["path/to/image1.jpg", "path/to/image2.jpg"]
prompt = "Compare these two images."

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(images)
)

output = generate(model, processor, formatted_prompt, images, verbose=False)
print(output)
```

#### 命令行

```sh
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Compare these images" --image path/to/image1.jpg path/to/image2.jpg
```

## 视频理解

MLX-VLM 还支持精选模型进行视频分析，如字幕生成、摘要等。

### 支持的模型

以下模型支持视频聊天：

1. Qwen2-VL
2. Qwen2.5-VL
3. Idefics3
4. LLaVA

更多模型即将推出。

### 使用示例

#### 命令行
```sh
mlx_vlm.video_generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "Describe this video" --video path/to/video.mp4 --max-pixels 224 224 --fps 1.0
```


这些示例演示了如何使用 MLX-VLM 进行多图像处理，以处理更复杂的视觉推理任务。

## 视觉特征缓存

在关于图像的多轮对话中，视觉编码器在每一轮都会运行，即使图像没有改变。`VisionFeatureCache` 将投影的视觉特征存储在以图像路径为键的 LRU 缓存中，因此昂贵的视觉编码器每个唯一图像只调用一次。

### 工作原理

1. **第一轮（缓存未命中）** -- `encode_image()` 运行完整的视觉流水线（视觉塔 + 投影器），将结果存储在缓存中，并将其传递给语言模型。
2. **后续轮（缓存命中）** -- 缓存的特征通过 `cached_image_features` 直接传递，完全跳过视觉编码器。
3. **图像切换** -- 当图像改变时，这是一个新的缓存键，因此计算特征并缓存。切换回之前的图像是缓存命中。

缓存最多容纳 8 个条目（可配置），并使用 LRU 淘汰。

### CLI

所有聊天界面自动使用 `VisionFeatureCache`：

```sh
# Gradio 聊天界面
python -m mlx_vlm.chat_ui --model google/gemma-4-26b-a4b-it

# 带有 Rich UI 的交互式聊天（使用 /image 命令加载图像）
python -m mlx_vlm.chat --model google/gemma-4-26b-a4b-it

# 内联聊天模式
python -m mlx_vlm.generate \
  --model google/gemma-4-26b-a4b-it \
  --image path/to/image.jpg \
  --chat \
  --max-tokens 200
```

### Python

```python
from mlx_vlm import load, stream_generate, VisionFeatureCache
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("google/gemma-4-26b-a4b-it")
cache = VisionFeatureCache()

image = "path/to/image.jpg"

# 第 1 轮 - 缓存未命中，编码图像
prompt1 = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
for chunk in stream_generate(model, processor, prompt1, image=[image],
                              max_tokens=200, vision_cache=cache):
    print(chunk.text, end="")

# 第 2 轮 - 缓存命中，跳过视觉编码器
prompt2 = apply_chat_template(processor, model.config, "What colors do you see?", num_images=1)
for chunk in stream_generate(model, processor, prompt2, image=[image],
                              max_tokens=200, vision_cache=cache):
    print(chunk.text, end="")
```

### 服务器

服务器自动在相同图像的多个请求中缓存视觉特征。无需配置 -- 缓存在加载模型时创建，并在卸载时清除。

```sh
mlx_vlm.server --model google/gemma-4-26b-a4b-it
```

通过 `/v1/chat/completions`（流式和非流式）和 `/responses` 进行的多轮对话都受益。在多个请求中发送的相同图像将只编码一次。

### 性能

在 `google/gemma-4-26b-a4b-it` 上经过 10 轮多轮对话测试：

| 指标 | 无缓存 | 有缓存 |
|--------|--------------|------------|
| 提示 TPS | ~48 | ~550-825 |
| 加速 | -- | **11x+** |
| 峰值内存 | 52.66 GB | 52.66 GB（平坦） |

生成速度（~31 tok/s）和内存不受影响 -- 只有提示处理变得更快。

## TurboQuant KV 缓存

TurboQuant 在生成期间压缩 KV 缓存，从而在保持质量的同时，用更少的内存实现更长的上下文长度。

### 快速入门

```sh
# 3.5 位 KV 缓存量化（3 位键 + 4 位值）
mlx_vlm generate \
  --model mlx-community/Qwen3.5-4B-4bit \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant \
  --prompt "Your long prompt here..."
```

```python
from mlx_vlm import generate

result = generate(
    model, processor, prompt,
    kv_bits=3.5,
    kv_quant_scheme="turboquant",
    max_tokens=256,
)
```

```sh
# 使用 TurboQuant 的服务器
mlx_vlm server \
  --model google/gemma-4-26b-a4b-it \
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

### 工作原理

TurboQuant 使用随机旋转 + 代码本量化（[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)）将 KV 缓存条目从 16 位压缩到每维 2-4 位：

- **键和值**：带有 Hadamard 旋转的 MSE 代码本量化
- **分数位**（例如 3.5）：键使用较低的位，值使用较高的位（3 位 K + 4 位 V）

自定义 Metal 内核直接在打包的量化数据上融合分数计算和值聚合，避免了完全反量化。

### 性能

在 Qwen3.5-4B-4bit 的 128k 上下文上测试：

| 指标 | 基线 | TurboQuant 3.5 位 |
|--------|----------|-------------------|
| KV 内存 | 4.1 GB | 0.97 GB（**减少 76%**） |
| 峰值内存 | 18.3 GB | 17.3 GB（**-1.0 GB**） |

在 512k+ 上下文中，由于减少的内存带宽要求，TurboQuant 的每层注意力**比 FP16 SDPA 更快**。

在 gemma-4-31b-it 的 128k 上下文上测试：

| 指标 | 基线 | TurboQuant 3.5 位 |
|--------|----------|-------------------|
| KV 内存 | 13.3 GB | 4.9 GB（**减少 63%**） |
| 峰值内存 | 75.2 GB | 65.8 GB（**-9.4 GB**） |

### 支持的位宽

| 位 | 压缩 | 最适合 |
|------|------------|----------|
| 2 | ~8x | 最大压缩，一些质量损失 |
| 3 | ~5x | 质量和压缩的良好平衡 |
| 3.5 | ~4.5x | 推荐默认（3 位键 + 4 位值） |
| 4 | ~4x | 最佳质量，适度压缩 |

### 兼容性

TurboQuant 自动量化 `KVCache` 层（全局注意力）。具有 `RotatingKVCache`（滑动窗口）或 `ArraysCache`（MLA/吸收键）的模型为这些层保留其本机缓存格式，因为它们已经是内存高效的。

# 微调

MLX-VLM 支持使用 LoRA 和 QLoRA 微调模型。

## LoRA & QLoRA

要了解更多关于 LoRA 的信息，请参阅 [LoRA.md](./mlx_vlm/LORA.MD) 文件。
