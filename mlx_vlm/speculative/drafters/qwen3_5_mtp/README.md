# Qwen3.5 / Qwen3.6 / Qwen3.8 MTP Drafter

MLX support for Qwen3.5, Qwen3.6, and Qwen3.8 native Multi-Token Prediction
(MTP) drafters used by the speculative decoding path.

## What it is

These checkpoints can include native `mtp.*` weights in the target model. The
CLI and server can load those weights directly from the original checkpoint:

```bash
python -m mlx_vlm.generate \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --draft-model nvidia/Qwen3.8-27B-NVFP4 \
  --draft-block-size 3 \
  --prompt "Write a quicksort in Python." \
  --max-tokens 256 --temperature 0

python -m mlx_vlm.server \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --draft-model nvidia/Qwen3.8-27B-NVFP4 \
  --draft-block-size 3
```

`--draft-kind mtp` is auto-detected. Only the bundled MTP tensors are retained
for the drafter; no extracted checkpoint is written. This also accepts local
checkpoint directories. A converted target that dropped MTP tensors must use
the matching original checkpoint or a standalone MTP folder as `--draft-model`.

Programmatically, use `load_drafter(original_checkpoint)` to get the same behavior.

On a running server, enable bundled MTP through the settings endpoint:

```bash
curl -X PATCH http://localhost:8080/v1/settings \
  -H 'Content-Type: application/json' \
  -d '{"spec_draft_model":"nvidia/Qwen3.8-27B-NVFP4","spec_draft_kind":"mtp"}'

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

These are server-wide settings, not per-request generation fields. The text
model reloads with its drafter on the next generation request; a server restart
is unnecessary. Both `/v1/chat/completions` and `/v1/responses` use the configured
drafter. If the server requires an API key, include its bearer authorization
header on these requests.

Optionally, split those weights into a standalone drafter folder with:

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
- Official block-FP8 MTP weights are converted to MLX MXFP8 when loading or splitting.
- Multimodal prompts are supported, but image/video prefill still runs through
  the target model; MTP accelerates the text decode tail.
