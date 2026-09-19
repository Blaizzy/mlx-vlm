# Mistral Large 3

Vision-language support for **Mistral Large 3** (675B, MoE).

|  |  |
| --- | --- |
| **Model ID** | `mistralai/Mistral-Large-3-675B-Instruct-2512` |
| **Architecture** | Pixtral vision encoder + patch-merge projector + DeepSeek-V3-style (MLA + MoE) language backbone |

## Composition

This model reuses existing mlx-vlm components rather than reimplementing them:

- **Vision:** `pixtral` `VisionModel` (patch conv, RoPE 2D, transformer).
- **Projector:** `mistral3` `Mistral3MultiModalProjector` (RMSNorm -> patch merger -> two linears), which matches Mistral Large 3's `pre_mm_projector_norm` / `patch_merger` / `vision_language_adapter`.
- **Language:** `deepseek_v3` `LanguageModel` (multi-head latent attention + routed/shared experts).

`sanitize` translates the Mistral-native tensor names (`wq_a`, `wkv_b`, `experts.N.w1`, `vision_encoder.*`, `vision_language_adapter.w_in`, ...) to the layouts those components expect, then defers expert stacking and the `kv_b` -> `embed_q`/`unembed_out` split to `deepseek_v3`'s sanitize.

## Checkpoint conversion

The published checkpoint ships Mistral's native format (`params.json` + `tekken.json`, no `config.json`) with block-FP8 weights. `config.config_from_params` maps `params.json` to an mlx-vlm config. Convert the weights to an mlx-vlm 4-bit build (~340 GB) before loading; loading the full 675B model requires a machine with enough memory (a 512 GB Apple silicon machine fits the 4-bit build).
