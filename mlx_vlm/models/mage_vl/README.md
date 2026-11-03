# Mage-VL

Mage-VL is a codec-native vision-language model from Microsoft supporting text, image, and video inputs. It combines a Mage-ViT visual encoder with a Qwen3 language backbone.

## Model

- [nativ-community/Mage-VL-OptiQ-4bit](https://huggingface.co/nativ-community/Mage-VL-OptiQ-4bit)
- [microsoft/Mage-VL-Cowork](https://huggingface.co/microsoft/Mage-VL-Cowork)

## Generate

Image:

```sh
mlx_vlm.generate \
  --model nativ-community/Mage-VL-OptiQ-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Video:

```sh
mlx_vlm.generate \
  --model nativ-community/Mage-VL-OptiQ-4bit \
  --video /path/to/video.mp4 \
  --prompt "Summarize this video." \
  --max-tokens 256
```

## Mage-VL-Cowork

Mage-VL-Cowork is Microsoft's 4B computer-use model, with the same Mage-ViT
encoder and Qwen3 decoder. Its source checkpoint declares `llava_onevision2`
and stores the vision weights under `visual.*`; both are handled natively by
the Mage-VL implementation, without executing Hugging Face remote code.

For computer-use prompts, copy the upstream
[`MAGE_HYBRID_SYSTEM_PROMPT`](https://huggingface.co/microsoft/Mage-VL-Cowork/blob/main/osworld_eval/mm_agents/mage_hybrid_agent.py)
into `mage_cowork_system.txt`. Run the original BF16 checkpoint on a screenshot:

```sh
mlx_vlm.generate \
  --model microsoft/Mage-VL-Cowork \
  --image /path/to/screenshot.png \
  --system "$(cat mage_cowork_system.txt)" \
  --prompt "Click the Save button." \
  --enable-thinking \
  --temperature 0 \
  --max-tokens 4096
```

Convert it to 4-bit:

```sh
mlx_vlm.convert \
  --hf-path microsoft/Mage-VL-Cowork \
  --mlx-path ./Mage-VL-Cowork-4bit \
  -q --q-bits 4 --q-group-size 64
```

The system prompt defines the `computer_use` and `bash` tools and the
model's `<think>`, `Action:`, and JSON `<tool_call>` response format. GUI
coordinates use the range 0–999. Pass the screenshot with the user message;
subsequent turns can include previous screenshots, assistant calls, and tool
results through `apply_chat_template` with `enable_thinking=True` (or
`--enable-thinking` in the CLI), matching the upstream inference setup. Allow
up to 4096 output tokens for agent turns. MLX-VLM generates the calls; the
caller supplies their execution and returns tool results. A response can request
another screenshot before attempting the requested action.
