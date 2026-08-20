---
name: add-new-model
description: Use this skill when the user wants to add or port a new model architecture to MLX-VLM — mapping a Hugging Face model_type to a new file under mlx_vlm/models, writing the ModelConfig, matching layer/weight names, reusing a similar existing model, adding a test class, and validating the port. Covers vision-language, language, audio, and diffusion model ports.
---

# Add a New Model

Use this workflow to port a Hugging Face model to MLX-VLM.

## Layout Rules

- New model lives in `mlx_vlm/models/<model_type>/`, and the main file is named after the `config.json` `model_type` (e.g. `model_type: "llava"` → `mlx_vlm/models/llava/llava.py`). The loader resolves the arch by importing `mlx_vlm.models.<model_type>` (see `MODEL_REMAPPING` in `mlx_vlm/utils.py` for aliases).
- Split by concern like the existing families: `language.py`, `vision.py`, `config.py`, `processing_*.py`. A new kernel/helper goes in **its own file** in the model dir, not inside `language.py`.
- **Start from a similar existing model** in `mlx_vlm/models/` and adapt — don't write from scratch.

## Steps

1. **Confirm weights are safetensors.** If not, convert them first (HF safetensors converter), then proceed.
2. **Copy a close relative** as scaffolding (same attention/vision style). Rename to the new `model_type`.
3. **Write `config.py`.** A `ModelConfig` dataclass; give every new field a **backward-compatible default** (`None`/`0`/`False`) so existing configs still load unchanged. Add inline `# comments`.
4. **Map layer/weight names.** Determine them by one of:
   - the Transformers implementation, if you know it;
   - loading the weights and printing key names;
   - reading `model.safetensors.index.json` in the HF repo.
5. **Wire the forward pass** (embeddings → vision/audio encoder → projector → language model), reusing shared helpers (`prompt_utils.py`, processors) where possible.
6. **Convert to MLX** to get loadable weights — `Skill("mlx-vlm-skills:convert-quantize")`.
7. **Add a test class** in `mlx_vlm/tests/test_models.py` (e.g. `TestMyModel`) — a tiny random-weight config, a shape/forward check, and (if applicable) an exactness check against a reference path in the degenerate limit. Do not create a standalone test file.

## Determine Layer Names (quick)

```bash
uv run python - <<'PY'
from huggingface_hub import hf_hub_download
import json
p = hf_hub_download("<repo>", "model.safetensors.index.json")
print("\n".join(sorted(json.load(open(p))["weight_map"])[:60]))
PY
```

## Validation

- Run the model's test class:
  ```bash
  cd mlx_vlm && uv run --with pytest python -m pytest tests/test_models.py -q -k "TestMyModel"
  ```
- Then a real end-to-end generation via `Skill("mlx-vlm-skills:cli-inference")`. Run on Apple Silicon with enough RAM for the model; isolate a submodule with random-init weights if the full model does not fit (do not run large models on an 8 GB machine).
- Compare a few greedy outputs against the reference implementation (Transformers) on the same prompt to confirm correctness, not just that it runs.
- Format with pre-commit and follow PR expectations — `Skill("mlx-vlm-skills:contributing")`.

## Common Failure Routing

- `Model type <x> not supported`: the folder name / `model_type` don't match, or the arch import failed — check `mlx_vlm/models/<model_type>/` exists and imports cleanly.
- Weight-name mismatches on load: your module attribute names don't match the checkpoint; reconcile against `model.safetensors.index.json`.
- Processor/chat-template errors: mirror a sibling model's `processing_*.py` and `prompt_utils` usage.
- If you get stuck and want to file it, use `Skill("mlx-vlm-skills:reproducible-github-issues")`.
