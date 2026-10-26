# LFM3 preview models

Native MLX support for `lq-preview/LFM3-1.7B-Preview` and
`lq-preview/LFM3-VL-1.9B-Preview`. These checkpoints require Hugging Face access.
Loading and inference use the native implementation and do not require
`trust_remote_code=True` or PyTorch. Checkpoint licenses continue to apply to
the weights. The implementation adapts the Apache-2.0 reference code supplied
with the vision checkpoint; see [LICENSE](LICENSE).

## Usage

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("lq-preview/LFM3-1.7B-Preview")
prompt = apply_chat_template(
    processor, model.config, "What is 17 + 25?",
    enable_thinking=False,
)
print(generate(model, processor, prompt, max_tokens=128, temperature=0).text)

model, processor = load("lq-preview/LFM3-VL-1.9B-Preview")
prompt = apply_chat_template(
    processor, model.config, "Describe this image.",
    num_images=1, enable_thinking=False,
)
print(generate(
    model, processor, prompt, image=["image.png"],
    max_tokens=128, temperature=0,
).text)
```

`LiquidAI/d1-2B` shares the LFM3-VL `model_type` but is a decision model, not
a chat model; loading it raises an error. Run it with Transformers
(`trust_remote_code=True`, `model.system_one(...)`) as its model card shows.

The vision model accepts multiple images and pre-sampled video frames. The
standard generation API decodes video paths and supplies the sampled FPS;
direct processor calls should supply `fps` or `video_metadata` to preserve
timestamps. Images use the checkpoint's resolution budget, with optional
`min_pixels` and `max_pixels` processor overrides. NumPy/PIL preprocessing
avoids a PyTorch runtime dependency; resized images can differ by one 8-bit
pixel level from the reference torchvision bicubic resize.

Standard conversion quantizes the language backbone and retains the vision
tower's original precision:

```sh
python -m mlx_vlm convert \
  --hf-path lq-preview/LFM3-VL-1.9B-Preview \
  --mlx-path ./LFM3-VL-1.9B-Preview-4bit \
  -q --q-bits 4 --q-group-size 64
```

## Architecture and validation

The shared text backbone combines gated short convolutions, full attention,
and sliding attention. Full attention uses partial RoPE across matching pairs
in the two head halves; sliding attention rotates the entire head. Attention
has a sigmoid gate per query head. Sandwich normalization is checkpoint
configurable. Sliding caches retain their absolute rotary positions after
wrapping, including merged batches with different prompt lengths.

The vision tower uses a linear temporal patchifier, gated spatial depthwise
convolutions, and frame-local bidirectional attention with axial RoPE. The
projector normalizes each patch before merging spatial blocks. Weight layout
conversion is idempotent for both original and converted checkpoints.

Validated on an Apple M3 Ultra with MLX 0.32.4 and Transformers 5.20.0:

| Checkpoint | Pinned revision | Pretrained float32 max logit error |
| --- | --- | --- |
| LFM3-1.7B-Preview | `8ab5fc0470627fad4cec57a0fa96cf58a91a033f` | `1.08e-4` |
| LFM3-VL-1.9B-Preview | `641d9dc29b6a1bdc66b800fc4d53ac1117344608` | `5.20e-5` |

These errors compare final prompt logits against the supplied PyTorch models
using identical input tensors, with matching next-token argmax. Tiny models
also match all reference logits within `3e-7`. BF16 inference answered `42`,
read `CODE 5739`, and identified the left red shape and right blue circle.
These are smoke checks, not a comprehensive quality benchmark.

Regression coverage lives in `test_models.py`, `test_processors.py`, and
`test_model_ops.py` under `mlx_vlm/tests`, with tiny configs in `model_cases.json`.
It includes mixed caches, sliding-window wraparound, batch offsets, partial
rotary coordinates, frame separation, placeholder counts, and quantized
checkpoint round trips.

LFM3-Omni is not included: its checkpoint does not supply reference code for
the audio encoder, depth model, and LiquidCodec.
