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
| `test_models.py` + `model_cases.json` | Shared language, vision, projector, embedding, position, and native forward/cache contracts |
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
larger `test_processors.py`, `test_generate.py`, `test_server.py`, and
`test_speculative.py` contain their existing broad integration checks; related
small files should not automatically be added to those large modules.

## JSON model cases

`model_cases.json` contains 45 configurable contract cases and 17 native
forward/cache cases. `test_models.py` collects each case separately with a stable
model ID, so failures can be selected with `pytest -k`:

```sh
python -m pytest -q mlx_vlm/tests/test_models.py -k llava_bunny
```

For a contract case, `module` names the module under `mlx_vlm.models`, `config`
contains ordinary nested model settings, and `checks` lists check names in order:

```json
{
  "id": "tiny_mistral_language",
  "module": "mistral3",
  "config": {
    "text_config": {
      "model_type": "mistral",
      "hidden_size": 16,
      "intermediate_size": 32,
      "num_hidden_layers": 1,
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
      "head_dim": 8,
      "rms_norm_eps": 0.00001,
      "rope_theta": 10000.0,
      "vocab_size": 32
    },
    "vision_config": {
      "hidden_size": 16,
      "intermediate_size": 32,
      "num_hidden_layers": 1,
      "num_attention_heads": 2
    },
    "model_type": "mistral3"
  },
  "checks": ["language", "input_embeddings"]
}
```

The runner constructs the family's `ModelConfig`, nested `TextConfig`/`VisionConfig`
and other config classes, then calls `Model(config)`. Python selects check
arguments and reads dimensions from the config. Every case gets fresh config
objects and a fresh model; JSON contains no constructor calls or object references.

Most cases need no wiring overrides. `vision_path` and `projector_path` select
unusual component locations; defaults are `vision_tower` and
`multi_modal_projector`. `language_only: true` constructs
`LanguageModel(config.text_config, config)` for the isolated Qwen language check.
The optional `vision` object holds input data and layout settings: `input_shape`,
`feature_layer`, `channel_first`, and `grid_thw` (integer grid by default;
`grid_dtype: "float32"` preserves the floating-grid scenario).

The `dense` table retains the prototype's name and includes both dense and MoE
families. Each entry checks a full forward pass and cached token decode.
Model-specific regression scenarios need explicit assertions in the relevant
domain test module. The shared contracts do not replace numerical-reference,
checkpoint-conversion, or stateful integration assertions; MoE offload remains in
`test_moe_offload.py`, and training gradients remain in `test_trainer.py`.

These files retain separate execution boundaries:

- `test_deepseek_v4_reference.py` requires an external checkpoint and reference
  fixture and is the documented entry point for official-reference parity.
- `test_nemotron_voicechat_dependency_floor.py` checks the dependency floor in a
  fresh interpreter and supports direct script execution.
- `test_gemma4_assistant_masks_static.py` loads mask code with fake dependencies
  without importing MLX in the test module.
- `test_smoke.py` is the manual model-download runner and is excluded from CI's
  automated suite.
