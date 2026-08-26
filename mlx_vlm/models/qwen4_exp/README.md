# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next is a large multimodal mixture-of-experts model from Qwen for
text, image, and video understanding. The checkpoint uses the experimental
`qwen4_exp` architecture, combining Gated DeltaNet layers, Qwen Sparse
Attention (QSA), hashed n-gram PLE embeddings, hyper-connections, and a
Qwen3-style vision encoder.

## Model

- Hugging Face ID: `Qwen/Qwen3.8-Flash-Next`
- Modalities: text, image, and video
- Architecture: 48-layer hybrid DeltaNet/QSA MoE with 512 experts
- Best for: multimodal chat, visual reasoning, document and image analysis,
  and video understanding

## CLI

Text generation:

```sh
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --prompt "Explain sparse attention in one paragraph." \
  --max-tokens 256
```

Image understanding:

```sh
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --image ./image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Video understanding:

```sh
mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --video ./video.mp4 \
  --fps 1.0 \
  --prompt "Summarize this video." \
  --max-tokens 256
```

## Python

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model_path = "Qwen/Qwen3.8-Flash-Next"
model, processor = load(model_path)

images = ["./image.jpg"]
prompt = apply_chat_template(
    processor,
    model.config,
    "What is happening in this image?",
    num_images=len(images),
)

result = generate(
    model=model,
    processor=processor,
    prompt=prompt,
    image=images,
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
```

## Optional quantization

The official BF16 checkpoint is approximately 360 GB. Depending on the
available memory and desired quality/performance tradeoff, it can also be
converted to a lower-bit MLX checkpoint. For example:

```sh
mlx_vlm.convert \
  --hf-path Qwen/Qwen3.8-Flash-Next \
  --mlx-path ~/Qwen3.8-Flash-Next-3bit \
  --quantize \
  --q-group-size 32 \
  --q-bits 3
```

Group size 32 allows the PLE embedding dimensions to be quantized. The bit
width and output path can be adjusted for the target hardware.

## Notes

- The upstream checkpoint includes a separate MTP predictor; the base
  conditional-generation runtime loads the text and vision model and ignores
  the MTP tensors.
- QSA maintains an auxiliary index-key cache in addition to the normal KV
  cache. Single-request generation, chunked prefill, and uniform KV-cache
  quantization are supported.
- The current QSA cache path is not yet wired into continuous batching.
- Long image or video prompts may benefit from a smaller
  `--prefill-step-size` to reduce peak memory.
