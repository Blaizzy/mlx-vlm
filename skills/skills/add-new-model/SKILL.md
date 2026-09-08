---
name: add-new-model
description: Use this skill when the user wants to add or port a new model architecture to MLX-VLM — mapping a Hugging Face model_type to a new file under mlx_vlm/models, writing the ModelConfig, matching layer/weight names, reusing a similar existing model, adding a test class, and validating the port. Covers vision-language, language, audio, and diffusion model ports.
---

# Add a New Model

Use this workflow to port a new model to MLX-VLM.

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
6. **Convert to MLX** to get loadable weights — `Skill("mlx-vlm-skills:convert-quantize")`. Upload them to to `mlx-community` (which is self-add) on HF if there is no usable official repo. If the model is already usable directly from HF, you can skip this step and just use the repo directly.
7. **Check tool calling for every model.** Inspect the official model card, chat template, and reference parser to establish support, including for models that reuse an existing architecture. If supported, complete [Tool Calling](#tool-calling); otherwise record whether support is absent or still unknown.
8. **Add a test class** in `mlx_vlm/tests/test_models.py` (e.g. `TestMyModel`) — a tiny random-weight config, a shape/forward check, and (if applicable) an exactness check against a reference path in the degenerate limit. Do not create a standalone test file.
9. **Add a README in the model dir** with a short description of the model, supported HF repos, and example usage.

## Tool Calling

- Verify `_infer_tool_parser_from_processor` against the model's actual chat template. Reuse a compatible parser in `mlx_vlm/tool_parsers/`; add a parser and register its template markers in `__init__.py` only when needed. Check that existing formats still select the right parser.
- Match the native protocol and schema types, including its string escaping or CDATA rules where applicable. Exercise `process_tool_calls` in `mlx_vlm/server/responses_state.py`: it strips the start/end markers before invoking the parser. Confirm tools and prior tool results render correctly through the chat template.
- Keep regressions compact in `TestProcessToolCalls` in `mlx_vlm/tests/test_server.py`, reusing existing streaming tests and fixtures. Avoid a new model-specific test file or duplicated endpoint mocks. Cover a single call, argument types, malformed input, and ordinary text around calls.
- For models supporting multiple calls per response, check both different functions and repeated calls to the same function with different arguments. Assert independent arguments, unique IDs, and ordered indices. Verify split streaming markers do not leak markup or swallow surrounding text; the client controls concurrent tool execution.
- Run the focused parser cases and shared streaming tests from the repository root: `uv run --with pytest python -m pytest mlx_vlm/tests/test_server.py mlx_vlm/tests/test_responses_state.py -q -k 'ToolCall or tool_content'`. Include existing tests for any reused parser that changed.
- With loadable weights, smoke-test tools through the server in streaming and non-streaming modes, plus a normal reply without a call. Report whether validation used actual generation or synthetic output. Include a concrete before/after response example when describing a tool-calling fix in a PR.

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
- Compare a few greedy outputs against the reference implementation (transformers / diffusers) on the same prompt to confirm correctness, not just that it runs.
- Format with pre-commit and follow PR expectations — `Skill("mlx-vlm-skills:contributing")`.
- Ensure all commits are [signed](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits) or else the opened PR will not be merged.

## Common Failure Routing

- `Model type <x> not supported`: the folder name / `model_type` don't match, or the arch import failed — check `mlx_vlm/models/<model_type>/` exists and imports cleanly.
- Weight-name mismatches on load: your module attribute names don't match the checkpoint; reconcile against `model.safetensors.index.json`.
- Processor/chat-template errors: mirror a sibling model's `processing_*.py` and `prompt_utils` usage.
- If you get stuck and want to file it, use `Skill("mlx-vlm-skills:reproducible-github-issues")`.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Use ASD-STE100 Simplified Technical English, simple wordings
