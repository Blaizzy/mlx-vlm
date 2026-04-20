# Usage

## Command Line Interface (CLI)

Generate output from a model:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --max-tokens 100 --temperature 0.0 --image http://images.cocodataset.org/val2017/000000039769.jpg
```

## Chat UI with Gradio

Launch the chat interface:

```bash
python -m mlx_vlm.chat_ui --model mlx-community/Qwen2-VL-2B-Instruct-4bit
```

## Python Script

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))
output = generate(model, processor, formatted_prompt, image, verbose=False)
print(output)
```

## Speculative Decoding (DFlash)

Speed up generation 2–3× using a lightweight drafter model that predicts multiple tokens per round, verified in parallel by the target model.

### CLI

```bash
python -m mlx_vlm.generate \
    --model Qwen/Qwen3.5-4B \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --prompt "Write a quicksort in Python." \
    --max-tokens 512 --temperature 0 --enable-thinking
```

Works with images too:

```bash
python -m mlx_vlm.generate \
    --model Qwen/Qwen3.5-4B \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --image examples/images/cats.jpg \
    --prompt "Describe this image." \
    --max-tokens 256 --temperature 0 --enable-thinking
```

### Python — Single Sequence

```python
from mlx_vlm import load
from mlx_vlm.generate import stream_generate
from mlx_vlm.speculative.drafters import load_drafter

model, processor = load("Qwen/Qwen3.5-4B")
drafter = load_drafter("z-lab/Qwen3.5-4B-DFlash")

for result in stream_generate(
    model, processor,
    prompt="Write a quicksort in Python.",
    max_tokens=512,
    temperature=0,
    draft_model=drafter,
    enable_thinking=True,
):
    print(result.text, end="", flush=True)

# Acceptance stats
print(f"\nAccepted {sum(drafter.accept_lens)/len(drafter.accept_lens):.1f} tokens/round")
```

### Python — Batch Generate

Process multiple prompts in parallel:

```python
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate import (
    _dflash_rounds_batch,
    _make_cache,
    generation_stream,
)
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_lm.sample_utils import make_sampler

model, processor = load("Qwen/Qwen3.5-4B")
drafter = load_drafter("z-lab/Qwen3.5-4B-DFlash")
tok = processor.tokenizer
lm = model.language_model
sampler = make_sampler(temp=0)
eos_id = tok.eos_token_id

prompts = [
    "Write a quicksort in Python.",
    "What is the capital of France?",
    "Explain hash tables in 3 sentences.",
]

# Tokenize and left-pad to uniform length
texts = [
    apply_chat_template(
        processor, model.config, p,
        num_images=0, num_audios=0, enable_thinking=True,
    )
    for p in prompts
]
encoded = [tok.encode(t) for t in texts]
max_len = max(len(e) for e in encoded)
padded = [[0] * (max_len - len(e)) + e for e in encoded]
input_ids = mx.array(padded, dtype=mx.int32)
B = len(prompts)

# Create batch-aware caches and prefill
prompt_cache = _make_cache(lm, [0] * B)
lm._position_ids = None
lm._rope_deltas = None

target_layer_ids = list(drafter.config.target_layer_ids)
out = lm(input_ids, cache=prompt_cache, capture_layer_ids=target_layer_ids)
hidden = mx.concatenate(out.hidden_states, axis=-1)
first_bonus = sampler(out.logits[:, -1:]).squeeze(-1)
mx.eval(first_bonus, hidden, out.logits)

# Generate — finished sequences are automatically removed from
# the batch and the drafter restarts for the new batch size.
tokens_per_seq = [[] for _ in range(B)]
for tok_list, _ in _dflash_rounds_batch(
    model, drafter, prompt_cache, hidden,
    first_bonus=first_bonus,
    max_tokens=256,
    sampler=sampler,
    token_dtype=mx.int32,
    stop_check=lambda seq_idx, token_id: token_id == eos_id,
):
    for i, t in enumerate(tok_list):
        if t is not None:
            tokens_per_seq[i].append(t)

# Decode results
for i in range(B):
    all_toks = [int(first_bonus[i].item())] + tokens_per_seq[i]
    print(f"--- {prompts[i]}")
    print(tok.decode(all_toks))
```

### Supported Models

| Target | Drafter | Notes |
|--------|---------|-------|
| `Qwen/Qwen3.5-4B` | `z-lab/Qwen3.5-4B-DFlash` | Text + image. ~2.5× speedup on code/reasoning. |

The drafter is loaded via the shared `load_model` path — any model with a `dflash_config` key in its HF config is automatically detected.

## Server (FastAPI)

```bash
python -m mlx_vlm.server
```

See `README.md` for a complete `curl` example.

