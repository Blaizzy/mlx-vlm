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

## MTP speculative decoding

The official checkpoint also contains a native multi-token prediction (MTP)
head. Extract it into a standalone draft model, then use it for speculative
decoding with the official model:

```sh
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP

mlx_vlm.generate \
  --model Qwen/Qwen3.8-Flash-Next \
  --draft-model ./Qwen3.8-Flash-Next-MTP \
  --draft-kind mtp \
  --prompt "Explain speculative decoding in one paragraph." \
  --max-tokens 128
```

The released checkpoint contains one MTP layer, so the default draft block is
one speculative token. `--draft-block-size` can chain the head for additional
draft tokens; the best value depends on the prompt and hardware.

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

The extracted MTP head can be quantized independently:

```sh
python -m mlx_vlm.split_mtp \
  --model Qwen/Qwen3.8-Flash-Next \
  --output ./Qwen3.8-Flash-Next-MTP-3bit \
  --q-group-size 32 \
  --q-bits 3
```

## Notes

- The base conditional-generation runtime ignores the embedded `mtp.*`
  tensors. `mlx_vlm.split_mtp` extracts them into the standalone draft model
  used by speculative decoding.
- QSA maintains an auxiliary index-key cache in addition to the normal KV
  cache. Single-request generation, chunked prefill, continuous batching, and
  uniform KV-cache quantization for single requests are supported.
- KV-cache quantization is unsupported with continuous QSA batching;
  requesting both raises an explicit error to preserve the QSA indexer state.
- Long image or video prompts may benefit from a smaller
  `--prefill-step-size` to reduce peak memory.
- MLX conversions published before #2032 folded `Qwen4ExpRMSNorm`'s `1 + w`
  offset into their saved norm gains, so loading one would apply the offset
  twice and generate noise. Such a checkpoint is now detected and the offset
  subtracted back out, with a warning. Re-converting from
  `Qwen/Qwen3.8-Flash-Next` avoids relying on that fixup, which cannot restore
  gains the conversion rounded away. Set `text_config.preshifted_norm_weights`
  to `true` or `false` to override the detection.
