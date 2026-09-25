# mlx_vlm.chat — interactive terminal chat

Multi-turn chat with a vision-language model directly in the terminal, rendered with a Rich UI. Images can be loaded mid-conversation with an in-chat command, so you can start talking and attach a picture whenever you need to.

## Synopsis

```sh
mlx_vlm.chat [OPTIONS]
```

The console script `mlx_vlm.chat` also works.

## Examples

Start a chat with a model:

```sh
mlx_vlm.chat --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
```

Start a chat, then load an image once you are at the prompt:

```sh
mlx_vlm.chat --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
# at the "You" prompt:
/image /path/to/photo.jpg
What is happening in this picture?
```

## Options

### Model & loading

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/idefics2-8b-chatty-4bit` | Path to the model or model identifier. |
| `--resize-shape` | `None` | Resize shape for the image (one or more ints). |

### Sampling & output

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | `0.0` | Temperature for sampling. |
| `--max-tokens` | `2048` | Maximum number of tokens to generate. |
| `--eos-tokens` | `None` | EOS tokens to add to the tokenizer (one or more). |
| `--skip-special-tokens` | `False` | Skip special tokens in the detokenizer. |
| `--verbose` / `--no-verbose` | `True` | Stream tokens as they are generated (use `--no-verbose` to disable). |

### Thinking

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-thinking` | `False` | Enable thinking in the chat template (templates using `thinking_mode` receive `thinking_mode='enabled'`). |
| `--thinking-mode` | `None` | Set the chat-template thinking mode when supported (`enabled`, `disabled`, `adaptive`). |
| `--thinking-budget` | `None` | Maximum number of thinking tokens before forcing end-of-thinking. |
| `--thinking-start-token` | `<think>` | Token that marks the start of a thinking block. |
| `--thinking-end-token` | `</think>` | Token that marks the end of a thinking block. |

### KV cache & quantization

| Flag | Default | Description |
|------|---------|-------------|
| `--max-kv-size` | `None` | Maximum KV size for the prompt cache. |
| `--kv-bits` | `None` | Number of bits to quantize the KV cache to. |
| `--kv-key-bits` | `None` | Override the TurboQuant key bit-width (defaults to `floor(--kv-bits)`). |
| `--kv-value-bits` | `None` | Override the TurboQuant value bit-width (defaults to `ceil(--kv-bits)`). |
| `--kv-key-scheme` | `None` | Override the KV quantization backend for keys only (`uniform`, `turboquant`). |
| `--kv-value-scheme` | `None` | Override the KV quantization backend for values only (`uniform`, `turboquant`). |
| `--kv-group-size` | `64` | Group size for uniform KV cache quantization. |
| `--kv-quant-scheme` | `uniform` | KV cache quantization backend (`uniform`, `turboquant`). |
| `--quantized-kv-start` | `5000` | Start index for the quantized KV cache. |

### Performance

| Flag | Default | Description |
|------|---------|-------------|
| `--prefill-step-size` | `2048` | Number of tokens to process per prefill step. |

## In-chat commands

Once the chat is running, type these at the `You` prompt. Any other input is treated as a question or comment about the current image.

| Command | Description |
|---------|-------------|
| `/image <path>` | Load a new image for discussion. |
| `/clear` | Clear conversation history. |
| `/help` | Show the available commands. |
| `/exit` | Exit the chat. |

## See also

[generate](generate.md) · [Gradio chat UI](index.md#gradio-chat-ui)
