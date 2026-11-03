# Flint

Native MLX support for the text-only BF16
[`RecursiveSI/flint-preview`](https://huggingface.co/RecursiveSI/flint-preview)
checkpoint. The original safetensors load directly; no conversion or remote
Python code is required.

## Usage

```bash
python -m mlx_vlm.generate \
  --model RecursiveSI/flint-preview \
  --prompt "Write a Python function that merges overlapping intervals." \
  --max-tokens 4096 \
  --temperature 1.0 --top-p 0.95 --top-k 20
```

For a reproducible load of the validated checkpoint:

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(
    "RecursiveSI/flint-preview",
    revision="78d534d13014a6128405febe0c7b450dd41859fc",
    trust_remote_code=False,
)
prompt = apply_chat_template(processor, model.config, "What is 17 * 23?")
result = generate(model, processor, prompt, max_tokens=256, temperature=0.0)
print(result.text)
```

The checkpoint's own chat template supports reasoning and tool messages. Keep
`reasoning_content` in assistant history. Tool arguments may be dictionaries or
JSON strings; the existing `glm47` parser is selected from Flint's XML markers.
The model can emit `<think>...</think>` before its answer, so the generation
budget includes reasoning tokens.

## Implementation

- 49 decoder layers: two dense layers and 47 MoE layers with sigmoid routing,
  six of 256 routed experts per token, and an ungated shared expert.
- Per-head QK RMSNorm and learned attention sinks.
- A 513-token rotating cache for sliding attention and a full cache for global
  attention. RoPE is applied only where the checkpoint configuration requests it.
- FP32 output projection, matching the checkpoint's reference serving setup.
- Strict loading of individual expert tensors and reloadable stacked MLX weights.

## Validation

Validated locally on October 21, 2026. The affected model, processor, cache, and
tool-streaming suites passed: 1,040 tests and 57 subtests, with one skipped test.

The complete 79.6 GB BF16 checkpoint was loaded and evaluated on an Apple M5 Max
with 128 GiB unified memory using MLX 0.32.4. Load peak allocation was 81.2 GB;
the operating system and other applications need additional memory.

Greedy generation returned `391` for `17 * 23` and produced an interval-merging
function that passed seven functional examples. Both generations reached a
configured stop token. Decode throughput was approximately 52 tokens/second in
these two smoke runs, not a repeated benchmark.

Tests cover an independent NumPy model oracle, hybrid and padded batch caches,
strict weight loading, quantized serialization, tokenizer preservation, and
reasoning/tool-message replay. Original checkpoint layers 0, 2, and 48 also
matched an independent PyTorch calculation using FP32 views of the BF16 weights
(maximum absolute error below `2.4e-5`).

A real 545-token coding continuation exercised the sliding-window boundary with
cached chunks of 256, 256, 32, and 1 tokens. Full and chunked BF16 evaluation had
the same top-five next tokens but different logits (maximum absolute difference
`1.85`, probability-distribution KL divergence `0.0106`). This validates cache
continuation, not bitwise BF16 equivalence. Full SGLang/CUDA output parity and
long-context answer quality have not been validated.
