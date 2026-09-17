# Test organization

Add regressions and reusable helpers to the module covering the same production
behavior. Import shared helpers from that module instead of adding separate
fixture files. Prefer a new file when a test needs an independent dependency,
checkpoint, or collection boundary. Keep fixture setup and hardware skips scoped
to the tests that need them.

Run the automated suite from the repository root:

```sh
python -m pytest -q mlx_vlm/tests --ignore=mlx_vlm/tests/test_smoke.py
```

Use a module path or `-k` to select a smaller group while developing.

| Test module | Scope |
| --- | --- |
| `test_models.py` + `model_cases.json` | Shared language, vision, audio, projector, embedding, position, and native forward/cache contracts |
| `test_image_generation_models.py` | Bonsai, Flux2, Ideogram4, Z-Image, ERNIE Image, and Mage Flow generation/editing, components, loading, and conversion |
| `test_diffusion_models.py` | LLaDA, Nemotron, and DiffusionGemma model/generation contracts, including sampling, prefill, caches, numerical parity, self-conditioning, vision, sanitization, and quantization policy |
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
| `test_qwen3_5.py` | Qwen3.5 patch layouts and ragged attention fallbacks |
| `test_nemotron_voicechat.py` | VoiceChat runtime, streaming, and checkpoint conversion |
| `test_processors.py` | Image/video/audio processors, including Mage VL timestamps, patch positions, and visual-embedding integration |
| `test_speculative.py` | Drafter loading and compatibility, generation parity, verification, cache transactions, and quantized speculative state across model families |
| `test_speculative_masks_static.py` | Gemma assistant mask offsets with fake dependencies, without importing MLX |
| `test_rope.py` | Rotary embeddings, multimodal position IDs, and batched offsets |
| `test_audio_generation.py` | Audio generation, loading, downmixing, and resampling |
| `test_server.py` | Chat/Responses/Anthropic APIs, image endpoints, batching/cancellation, runtime settings, and reranking |
| `test_server_audio.py` | HTTP audio endpoints and realtime voice sessions |
| `test_cli.py` | CLI arguments, diffusion display/visualizer behavior, detector display options, and CLI/library default parity |
| `test_prompt_utils.py` | Prompt construction and reasoning-template arguments |
| `test_trainer.py` | Training workflows, trainer utilities, MRoPE/gated-delta gradients, and MoE gradient/expert-replacement checks |
| `test_utils.py` | General loading/conversion utilities and local Python model files |

Other architecture-specific modules remain focused on their own models. The
larger `test_processors.py`, `test_generate.py`, and `test_server.py` contain
their existing broad integration checks. Drafter and speculative-decoding tests
belong in `test_speculative.py`: reuse its tiny target/config factories and
parameterize shared contracts with descriptive family IDs. Checkpoint I/O, MTP
setup, quantization format matrices, and repeated-decode references are shared
within the suite. `test_speculative.py` supplies fresh tiny language configs
for speculative and training tests. It also owns MiniMax
speculative rollback checks; unrelated model tests remain in their existing
modules. Cache/position, sampling-parity, and batched-mask checks run through
shared contract runners. See [coverage measurements](speculative_coverage.md)
for the retained coverage and remaining gaps from the compact rewrite.

`test_diffusion_models.py` owns the shared DiffusionGemma helpers for tiny configs,
model construction, tokenization, encoder recording, and stream calls. CLI, APC,
and loading tests import the helpers they need from that module.
Processor checks live in `test_processors.py`, server block streaming in
`test_server.py`, and generation-config loading in `test_utils.py`. Keep numerical
and vision-specific assertions in `test_diffusion_models.py`; its optional
Transformers reference check still requires the reference dependencies.

`test_processors.py` uses `SMOKE_PROCESSORS` and `_make_processor` for shared
image/text contracts across 20 processor families. Helpers for tokenization,
image outputs, temporary checkpoint configs, and loader mocks stay in this file.
Reuse the parameterized output, EOS-token, tiled-image, and timestamp contracts
when adding cases. Keep distinct media, batching, serialization, and threading
assertions explicit; related integration scenarios may share their setup. AutoProcessor routing checks assert the selected
loader and its returned object, and exercise real incomplete-checkpoint behavior.

`test_server.py` keeps its endpoint, streaming-result, and worker doubles in the
same file. Reuse `_endpoint` and the request/result builders for API checks;
`_worker_setup`, `_running`, and `_drain` exercise the real generation worker with
a recording batch implementation. Parameterize protocol and model variants while
keeping response assertions, cancellation, tokenizer locking, and cache lifecycle
checks explicit. No separate server fixture module is needed.

`test_image_generation_models.py` owns the six image-generation families and
keeps checkpoint writers, packed-pipeline doubles, and tiny quantization models
in the same file. Reuse the parameterized discovery, dimension, download,
conversion, and weight-loading checks with descriptive family IDs. Retain
family-specific prompt, guidance, position, VAE, and numerical assertions next to
the shared contracts. Image CLI routing and request forwarding live in
`test_generate.py`; image HTTP endpoints stay in `test_server.py`.

## JSON model cases

`model_cases.json` contains 51 configurable contract cases and 17 native
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

Combine regressions that use the same model configuration into one case's
`checks` list. Both `qwen3_5` and `qwen3_5_moe` check language and vision output
shapes and dtypes in float32 and float16, plus the text-only `input_embeddings`
contract (an `InputEmbeddingsFeatures` result with populated `inputs_embeds`).
Their tiny language configs include one linear-attention layer and one
full-attention layer. The `qwen3_5` case also covers request-owned positions,
chunked prefill, and decode-time RoPE deltas;
`qwen3_5_moe` retains its chunked-prefill regression. Checks that replace inner
modules with recording stubs run after checks that need the original model.
The `inkling` case checks language, vision, audio, and text-only input embeddings.

The `audio` check supports Inkling, Gemma 3n, Gemma 4, and Gemma 4 Unified.
Use a tiny `audio_config`; optional `audio: {"frames": 33, "lengths": [33, 13]}`
sets the input frame count and valid prefix lengths. Defaults are 32 frames
and two rows with lengths 32 and 16. Lengths control Gemma masks; Inkling's
tower accepts unmasked integer dMel IDs. Input construction and tower/projector
selection stay in Python. The check evaluates outputs with float32 and float16
weights, verifies finite values, shapes, and projection into text dimensions,
and checks Gemma encoder masks and zeroed padding or Unified's compacted token
count. Gemma 3n/4 currently return float32 with float16 inputs; this is asserted
explicitly. `input_embeddings` remains text-only and does not test audio-token
insertion.

The `deepseek_v4` and `qwen4_exp` cases use shared language, vision, text-only
input-embedding, and `forward_cache` checks; DeepSeek also checks its aligner
through `projector`. Their standalone regression modules are removed. The
Qwen4 transaction tests read their tiny config from the same JSON case.
`forward_cache` reuses the native full-forward and cached-decode shape checks.

The `minimax_m3_vl` case replaces its standalone regression module with the
existing language, vision, projector, text-only input-embedding, and
`forward_cache` checks. Its tiny language config includes dense and MoE layers;
vision uses four flattened patches with a 2×2 grid. The MiniMax speculative
rollback checks remain in `test_speculative.py`.

The `indic_ocr` case checks its Qwen3.5-based recognizer with the existing
language, vision, text-only input-embedding, and `forward_cache` checks. Its
tiny config includes linear and full-attention layers and four vision patches.
The standalone OCR pipeline tests are removed; shared model contracts do not
cover combined checkpoint loading, layout cleanup, cropping, or reconstruction.

Most cases need no wiring overrides. `vision_path` and `projector_path` select
unusual component locations; defaults are `vision_tower` and
`multi_modal_projector`. The optional `vision` object holds input data and layout
settings: `input_shape`, `feature_layer`, `channel_first`, `grid_hw`, and
`grid_thw` (integer grid by default;
`grid_dtype: "float32"` preserves the floating-grid scenario). An `input_shape`
with more than two dimensions specifies the complete tensor shape, such as
Inkling's `[patches, time, height, width, channels]`. Set `feature_layer: null`
when the vision tower returns its feature tensor directly. DeepSeek's `grid_hw`
gives the patch-grid height and width to its vision tower and aligner.

The `dense` table retains the prototype's name and includes both dense and MoE
families. Each entry checks a full forward pass and cached token decode.
Model-specific regression scenarios need explicit assertions in the relevant
domain test module. The shared contracts do not replace numerical-reference,
checkpoint-conversion, or stateful integration assertions; MoE offload remains in
`test_moe_offload.py`, and training gradients remain in `test_trainer.py`.

These files retain separate execution boundaries:

- `test_speculative_masks_static.py` loads mask code with fake dependencies
  without importing MLX in the test module.
- `test_smoke.py` is the manual model-download runner and is excluded from CI's
  automated suite.
