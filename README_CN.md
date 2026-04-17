# MLX-VLM

MLX-VLM 是一个基于 Apple MLX 框架的视觉语言模型（VLM）和多模态模型（支持音频和视频）推理与微调工具包，专为 Apple Silicon Mac 优化。

## 目录
- [安装](#安装)
- [使用](#使用)
  - [命令行工具 (CLI)](#命令行工具-cli)
  - [Gradio 聊天界面](#gradio-聊天界面)
  - [Python 脚本](#python-脚本)
- [服务端 (FastAPI)](#服务端-fastapi)
  - [本次迭代优化](#本次迭代优化)
  - [服务端选项](#服务端选项)
  - [可用接口](#可用接口)
- [视觉特征缓存](#视觉特征缓存)
- [TurboQuant KV 缓存](#turboquant-kv-缓存)
- [微调](#微调)

## 安装

```sh
pip install -U mlx-vlm
```

## 使用

### 命令行工具 (CLI)

```sh
# 文本生成
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --prompt "你好"

# 图像理解
mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --image /path/to/image.jpg --prompt "描述这张图片"

# 音频理解
mlx_vlm.generate --model mlx-community/gemma-3n-E2B-it-4bit --max-tokens 100 --prompt "描述你听到的内容" --audio /path/to/audio.wav
```

### Gradio 聊天界面

```sh
mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

### Python 脚本

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# 加载模型
model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
config = model.config

# 准备输入
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "描述这张图片。"

# 应用对话模板
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))

# 生成输出
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## 服务端 (FastAPI)

```sh
# 启动服务（延迟加载，首次请求时加载模型）
mlx_vlm.server --port 8080

# 预加载模型启动
mlx_vlm.server --model /path/to/local/model --port 8080

# 启用 KV 缓存量化
mlx_vlm.server --model /path/to/model --kv-bits 4 --port 8080
```

### 本次迭代优化

#### 异步流式输出（非阻塞事件循环）

服务端将同步的 MLX 生成器通过 `asyncio.run_in_executor` 放入线程池执行，避免阻塞事件循环。解决了大模型（30B+）在 prefill 阶段耗时数秒导致客户端超时断开、出现 `socket.send()` 异常的问题。

**修复前：** 同步 `for chunk in token_iterator` 完全冻结事件循环，客户端/代理在 prefill 期间超时断开。

**修复后：** 每次迭代将控制权交还事件循环，SSE 连接保持活跃。

#### RotatingKVCache 量化修复

使用 `RotatingKVCache` 的模型（如 Gemma 4）在启用 KV 缓存量化时不再崩溃（`NotImplementedError: RotatingKVCache Quantization NYI`）。量化器现在能递归检测 `RotatingKVCache` 条目（包括嵌套的 `CacheList`/tuple 结构）并跳过它们。

#### 默认图片分辨率限制

服务端对输入图片默认应用 1280x1280 像素的分辨率上限，防止高分辨率图片展开为百万级 visual tokens 导致显存溢出（OOM）。可通过以下方式覆盖：

```sh
# 环境变量
DEFAULT_RESIZE_SHAPE=2048,2048 mlx_vlm.server --model <model>

# 或在 API 请求体中指定
{"resize_shape": [2048, 2048], ...}
```

#### 模型缓存复用

服务端复用已预加载的模型，不再因客户端请求的模型名称与本地路径不匹配而尝试重新从 HuggingFace 下载。

#### 测试模型

| 模型 | 参数量 | 状态 | 备注 |
|------|--------|------|------|
| gemma-4-31b-it-mxfp4 | 31B | 流式输出正常 | RotatingKVCache 量化修复已验证 |
| Qwen3.6-35B-A3B-4bit | 35B MoE | 流式输出正常 | venv 中需安装 `torch` + `torchvision` |

### 服务端选项

| 参数 | 说明 |
|------|------|
| `--model` | 预加载模型路径（HF repo ID 或本地路径，可选） |
| `--host` | 监听地址（默认 `0.0.0.0`） |
| `--port` | 端口号（默认 `8080`） |
| `--kv-bits` | KV 缓存量化位数（如 `3.5` 用于 TurboQuant） |
| `--kv-quant-scheme` | KV 缓存量化后端（`uniform` 或 `turboquant`） |
| `--trust-remote-code` | 信任远程代码 |

### 可用接口

| 接口 | 说明 |
|------|------|
| `/models` `/v1/models` | 列出本地可用模型 |
| `/chat/completions` `/v1/chat/completions` | OpenAI 兼容的对话接口，支持图片、音频、文本 |
| `/responses` `/v1/responses` | OpenAI 兼容的 responses 接口 |
| `/health` | 健康检查 |
| `/unload` | 卸载当前模型 |

#### 请求示例

```sh
# 文本对话
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true,
    "max_tokens": 100
  }'

# 图像理解
curl -X POST "http://localhost:8080/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-VL-32B-Instruct-8bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "input_image", "image_url": "/path/to/image.jpg"}
      ]
    }],
    "stream": true,
    "max_tokens": 500
  }'
```

## 视觉特征缓存

在多轮图像对话中，视觉编码器每轮都会运行，即使图片没有变化。`VisionFeatureCache` 将投影后的视觉特征存储在 LRU 缓存中（按图片路径索引），昂贵的视觉编码只需执行一次。

**性能测试**（`google/gemma-4-26b-a4b-it`，10 轮多轮对话）：

| 指标 | 无缓存 | 有缓存 |
|------|--------|--------|
| Prompt TPS | ~48 | ~550-825 |
| 加速比 | -- | **11x+** |
| 峰值内存 | 52.66 GB | 52.66 GB（持平） |

## TurboQuant KV 缓存

TurboQuant 在生成过程中压缩 KV 缓存，以更少的内存支持更长的上下文。

```sh
# 3.5-bit KV 缓存量化
mlx_vlm.server --model google/gemma-4-26b-a4b-it --kv-bits 3.5 --kv-quant-scheme turboquant
```

**性能测试**（Qwen3.5-4B-4bit，128k 上下文）：

| 指标 | 基线 | TurboQuant 3.5-bit |
|------|------|-------------------|
| KV 内存 | 4.1 GB | 0.97 GB（**减少 76%**） |
| 峰值内存 | 18.3 GB | 17.3 GB（**减少 1.0 GB**） |

## 微调

MLX-VLM 支持 LoRA 和 QLoRA 微调，详见 [LoRA.md](./mlx_vlm/LORA.MD)。
