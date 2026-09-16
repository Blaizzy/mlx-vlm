# Test organization

Add regressions to the module covering the same production behavior and reuse
its fixtures. Prefer a new file when a test needs an independent dependency,
checkpoint, or collection boundary. Keep fixture setup and hardware skips scoped
to the tests that need them.

Run the automated suite from the repository root:

```sh
python -m pytest -q mlx_vlm/tests --ignore=mlx_vlm/tests/test_smoke.py
```

Use a module path or `-k` to select a smaller group while developing.

| Test module | Scope |
| --- | --- |
| `test_tool_parsers.py` | ATEM, Cohere, Gemma 4, GLM, Mistral, Pythonic parsing, and parser selection |
| `test_apc.py` | Cache lookup, semantic keys, lifecycle, trace logging, and diagnostics |
| `test_apc_adapters.py` | Cache adapters, component snapshots, and model cache-layout compatibility |
| `test_apc_storage.py` | Memory budgets, disk eviction, persistence, and block handles |
| `test_apc_prefix.py` | Exact and partial prefix reuse with dense and hybrid models |
| `test_apc_quantized.py` | APC integration with quantized cache formats |
| `test_apc_settings.py` | Server-facing APC settings and environment overrides |
| `test_kv_cache_quantization.py` | Quantized cache lifecycle, batching, and attention masks |
| `test_turboquant.py` | TurboQuant cache integration, batched attention, and value kernels |
| `test_weight_quantization.py` | FP8 and one-bit weight conversion and execution |
| `test_moe_offload.py` | MoE checkpoint repacking, expert offload, output parity, and failure handling |
| `test_qwen3_5.py` | Qwen3.5 patch layouts, MTP sanitization, and ragged attention fallbacks |
| `test_qwen4_exp.py` | Qwen4 model behavior, external PLE storage, and MTP drafting |
| `test_deepseek_v4_vision.py` | DeepSeek V4 vision behavior and checkpoint conversion |
| `test_nemotron_voicechat.py` | VoiceChat runtime, streaming, and checkpoint conversion |
| `test_mage_vl.py` | Mage VL video processing and position handling |
| `test_dflash_drafters.py` | DFlash2, Laguna, and Muse Glimmer drafter contracts |
| `test_rope.py` | Rotary embeddings, multimodal position IDs, and batched offsets |
| `test_audio_generation.py` | Audio generation, loading, downmixing, and resampling |
| `test_server_audio.py` | HTTP audio endpoints and realtime voice sessions |
| `test_cli.py` | CLI arguments, detector display options, and CLI/library default parity |
| `test_prompt_utils.py` | Prompt construction and reasoning-template arguments |
| `test_trainer.py` | Training workflows, trainer utilities, and MRoPE/gated-delta gradient regressions |
| `test_utils.py` | General loading/conversion utilities and local Python model files |

Other architecture-specific modules remain focused on their own models. The
larger `test_models.py`, `test_processors.py`, `test_generate.py`, `test_server.py`,
and `test_speculative.py` contain their existing broad integration checks; related
small files should not automatically be added to those large modules.

These files retain separate execution boundaries:

- `test_deepseek_v4_reference.py` requires an external checkpoint and reference
  fixture and is the documented entry point for official-reference parity.
- `test_nemotron_voicechat_dependency_floor.py` checks the dependency floor in a
  fresh interpreter and supports direct script execution.
- `test_gemma4_assistant_masks_static.py` loads mask code with fake dependencies
  without importing MLX in the test module.
- `test_smoke.py` is the manual model-download runner and is excluded from CI's
  automated suite.
