# Qwen3.5 VL

`qwen3_5` (dense) and `qwen3_5_moe` (mixture-of-experts) implement the Qwen3.5
vision-language architecture: a hybrid Qwen3.5 text backbone that interleaves
gated-delta linear attention with full attention, paired with a Qwen3-VL vision
tower and spatial-merge connector. The same architecture backs several published
checkpoints, including Ornith 1.5, so third-party fine-tunes load without model
changes.

## Models

| Model | Repository | Variant | Notes |
|---|---|---|---|
| Qwen3.6 35B-A3B | `Qwen/Qwen3.6-35B-A3B` | MoE (`qwen3_5_moe`) | Same architecture and MTP head |
| Ornith 1.5 9B | `ornith-ai/Ornith-1.5-9B` | dense (`qwen3_5`) | Vision-language, image and video |
| Ornith 1.5 35B-A3B | `ornith-ai/Ornith-1.5-35B-A3B` | MoE (`qwen3_5_moe`) | 256 experts, 8 active (~3B active) |


## Details

| | |
|---|---|
| **Architecture** | Hybrid Qwen3.5 text (gated-delta + full attention) + Qwen3-VL vision tower |
| **Modalities** | Text, image, video |
| **Vocabulary** | 248320 |
| **Speculative decoding** | Single-layer MTP head (`mtp` drafter), dense and MoE |
| **Official Cards** | [Ornith 1.5 9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B), [Ornith 1.5 35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |

## CLI Usage

```bash
python -m mlx_vlm.generate \
    --model ornith-ai/Ornith-1.5-9B \
    --image path/to/image.jpg \
    --prompt "Describe this image." \
    --max-tokens 200 \
    --temperature 0.0
```

## Python Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "ornith-ai/Ornith-1.5-9B"
model, processor = load(model_path)

image = "path/to/image.jpg"
prompt = "Describe this image."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=1,
)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=image,
    max_tokens=200,
    temperature=0.0,
)
print(result.text)
```

## Speculative Decoding

These checkpoints ship a single-layer MTP head. Extract it into a standalone
drafter and run speculative decoding with the `qwen3_5_mtp` drafter, which
supports both the dense and mixture-of-experts variants:

```bash
python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
    --model ornith-ai/Ornith-1.5-9B \
    --output ornith-1.5-9b-mtp

python -m mlx_vlm.generate \
    --model ornith-ai/Ornith-1.5-9B \
    --draft-model ornith-1.5-9b-mtp \
    --image path/to/image.jpg \
    --prompt "Describe this image." \
    --max-tokens 200 \
    --temperature 0.0
```

## Architecture

- **Vision**: Qwen3-VL vision tower with patch embedding and a spatial-merge
  connector that projects merged patches into the text hidden size.
- **Language**: Hybrid Qwen3.5 backbone alternating gated-delta linear-attention
  layers with full-attention layers; the MoE variant replaces the MLP with a
  routed expert block plus a shared expert.
- **Processor**: Reuses the Qwen3-VL processor and prompt format; image, video,
  and vision-start token ids are read from the model config.
