# Qwen3.5 / Qwen3.6 MTP Drafter

MLX support for Qwen3.5 and Qwen3.6 native Multi-Token Prediction (MTP)
drafters used by the speculative decoding path.

## What it is

Qwen3.5 and Qwen3.6 checkpoints can include native `mtp.*` weights in the
target model. Those weights are split into a standalone drafter folder with:

- `config.json` using `model_type: "qwen3_5_mtp"`
- `model.safetensors` containing only the sanitized MTP weights
- tokenizer files copied from the source model when present

At runtime, pass that folder as `--draft-model` with `--draft-kind mtp`.

## Split a Drafter

Run the splitter as a Python module:

```bash
uv run python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model Qwen/Qwen3.5-4B \
  --output ./Qwen3.5-4B-mtp
```

Useful options:

- `--revision REV` to split from a specific Hugging Face revision.
- `--block-size N` to override the default speculative block size.
- `--force-download` to refresh the source model from Hugging Face.

Programmatic use:

```python
from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import split_qwen3_5_mtp

split_qwen3_5_mtp(
    source="Qwen/Qwen3.5-4B",
    output="./Qwen3.5-4B-mtp",
)
```

## Generate

```bash
uv run mlx_vlm.generate \
  --model Qwen/Qwen3.5-4B \
  --draft-model ./Qwen3.5-4B-mtp \
  --draft-kind mtp \
  --draft-block-size 4 \
  --prompt "Make a program to find pi" \
  --max-tokens 256 --temperature 0
```

## Server

```bash
uv run mlx_vlm.server \
  --model Qwen/Qwen3.5-4B \
  --draft-model ./Qwen3.5-4B-mtp \
  --draft-kind mtp \
  --draft-block-size 4
```

## Notes

- Greedy decoding (`temperature=0`) uses exact target verification.
- The drafter is tied to the target family and tokenizer it was split from.
- Batched Qwen MTP uses uniform acceptance to keep the drafter cache aligned.
- Multimodal prompts are supported, but image/video prefill still runs through
  the target model; MTP accelerates the text decode tail.
