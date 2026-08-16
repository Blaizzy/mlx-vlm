---
name: convert-quantize
description: Use this skill when the user wants to convert a Hugging Face model to MLX or quantize/dequantize one with mlx_vlm.convert, including bits and group size, quant modes (affine, mxfp4, nvfp4, mxfp8), RTN vs AWQ, mixed-bit recipes, dtype casts, calibration (text or multimodal), local vs Hub paths, revisions, uploading to the Hub, and convert/quant errors.
---

# Convert & Quantize

Use this workflow for `mlx_vlm.convert` — turning a Hugging Face checkpoint into MLX format, optionally quantizing it.

## First Checks

1. Confirm the source: a Hugging Face repo id or a local path (`--hf-path`, alias `--model`).
2. Confirm the model family is supported in `mlx_vlm/models/` (a folder named after the `config.json` `model_type`). If not, this is a porting task — switch to `Skill("mlx-vlm-skills:add-new-model")`.
3. Verify current flags before finalizing: `uv run mlx_vlm.convert --help`.
4. Entry point is `mlx_vlm.convert`. `python -m mlx_vlm.convert` is deprecated; use `uv run mlx_vlm.convert ...` or `python -m mlx_vlm convert ...`.

## Command Patterns

Plain convert (no quantization), saves to `./mlx_model` by default:

```bash
uv run mlx_vlm.convert --hf-path <repo-or-path> --mlx-path ./out-mlx
```

4-bit affine quantization (RTN, the default method):

```bash
uv run mlx_vlm.convert --hf-path <repo-or-path> --mlx-path ./out-4bit -q --q-bits 4 --q-group-size 64
```

Other quant modes (`--q-mode` sets its own bit/group defaults):

```bash
# mxfp4 (group 32, 4 bit), nvfp4 (group 16, 4 bit), mxfp8 (group 32, 8 bit)
uv run mlx_vlm.convert --hf-path <repo-or-path> --mlx-path ./out-mxfp4 -q --q-mode mxfp4
```

Mixed-bit recipe (per-layer bit allocation, llama.cpp-style):

```bash
# recipes: mixed_2_6 mixed_3_4 mixed_3_5 mixed_3_6 mixed_3_8 mixed_4_6 mixed_4_8
uv run mlx_vlm.convert --hf-path <repo-or-path> --mlx-path ./out-mixed -q --quant-predicate mixed_3_6
```

AWQ (activation-aware, needs a calibration pass):

```bash
uv run mlx_vlm.convert --hf-path <repo-or-path> --mlx-path ./out-awq -q --quant-method awq \
  --calibration multimodal --calibration-data /path/to/media   # or --calibration text (default)
```

dtype cast only / dequantize:

```bash
uv run mlx_vlm.convert --hf-path <repo-or-path> --mlx-path ./out-bf16 --dtype bfloat16
uv run mlx_vlm.convert --hf-path <quantized-repo> --mlx-path ./out-fp -d   # dequantize
```

Upload the result to the Hub:

```bash
uv run mlx_vlm.convert --hf-path <repo> --mlx-path ./out -q --upload-repo <user>/<name>-mlx
```

## Key Facts

- **Multimodal modules are skipped from quantization by default** (`skip_multimodal_module`) — vision/audio towers stay full precision; only the language model is quantized. This is expected, not a bug.
- `--q-mode` choices: `affine` (default, group 64 / 4 bit), `mxfp4` (32/4), `nvfp4` (16/4), `mxfp8` (32/8). `--q-bits`/`--q-group-size` override the mode defaults.
- `--quant-method`: `rtn` (default, round-to-nearest) or `awq` (needs calibration; `--calibration text|multimodal`, optional `--calibration-data`).
- `-q`/`--quantize` and `-d`/`--dequantize` are mutually exclusive.
- `--dtype` (from `config.json` `torch_dtype` if unset) casts float weights; useful for shrinking fp32 → bf16 without quantizing.
- The converted folder gets the weights, copied `*.py`/`*.json`, the processor, a regenerated `config.json`, and a model card — it is ready for `mlx_vlm.generate` and the server.

## Validation

- After converting, load and run a tiny generation to prove the checkpoint works — see `Skill("mlx-vlm-skills:cli-inference")`. Run on Apple Silicon with enough RAM for the model; do not attempt a large model on an 8 GB machine.
- For quality, compare a few greedy outputs between the source and the quantized model; large drops usually mean too-low bits or a wrong `--q-mode` for the model.
- For code changes touching conversion, run `uv run --with pytest python -m pytest mlx_vlm/tests/test_utils.py -q` and any model-specific test.
- If the outcome is a bug report, switch to `Skill("mlx-vlm-skills:reproducible-github-issues")`.
