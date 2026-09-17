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
| `test_models.py` + `model_cases.json` | Shared language, vision, audio, projector, embedding, position, native forward/cache contracts, and cached-image source checks |
| `test_image_generation_models.py` | Bonsai, Flux2, Ideogram4, Z-Image, ERNIE Image, and Mage Flow generation/editing, components, loading, and conversion |
| `test_diffusion_models.py` | LLaDA, Nemotron, and DiffusionGemma model/generation contracts, including sampling, prefill, caches, numerical parity, self-conditioning, vision, sanitization, and quantization policy |
| `test_tool_parsers.py` | Automatic discovery of all 13 parser modules, shared parsing/extraction/selection contracts, and format-specific edge cases |
| `test_apc.py` | Cache lookup, semantic keys, adapters, model compatibility, exact/partial prefix reuse, quantized checkpoints, memory budgets, disk persistence, trace logging, and diagnostics |
| `test_cache.py` | Cache lifecycle, recurrence, quantization, batching, attention masks, and vision-feature LRU behavior |
| `test_turboquant.py` | TurboQuant cache integration, batched attention, and value kernels |
| `test_weight_quantization.py` | FP8 and one-bit weight conversion and execution |
| `test_moe_offload.py` | MoE checkpoint repacking, expert offload, output parity, and failure handling |
| `test_attention.py` | Absorbed MLA gates and numerical parity, Qwen3.5 ragged decode fallbacks, and PaddleOCR vision fast paths |
| `test_audio_models.py` | MiniCPMO TTS, Qwen3 Omni, Nemotron Omni, and VoiceChat components, speech generation, streaming, and checkpoint conversion |
| `test_video_generation_models.py` | MiniMax H3 packing, components, conditioning workflows, cached trajectories, numerical references, and conversion |
| `test_video_generation.py` | Video model discovery, request/result adapters, progress, and audio/video muxing |
| `test_processors.py` | Image/video/audio processors, including Mage VL timestamps, patch positions, and visual-embedding integration |
| `test_speculative.py` | Drafter loading and compatibility, generation parity, verification, cache transactions, and quantized speculative state across model families |
| `test_rope.py` | Rotary embeddings, multimodal position IDs, and batched offsets |
| `test_audio_generation.py` | Audio generation, loading, downmixing, and resampling |
| `test_server.py` | Chat/Responses/Anthropic APIs, image endpoints, batching/cancellation, runtime settings, reranking, Responses normalization, tool stream state, HTTP audio, and realtime voice sessions |
| `test_cli.py` | Text/image/audio/video CLI routing, arguments, diffusion display/visualizer behavior, detector display options, and CLI/library default parity |
| `test_prompt_utils.py` | Prompt construction and reasoning-template arguments |
| `test_trainer.py` | Training workflows, trainer utilities, MRoPE/gated-delta gradients, and MoE gradient/expert-replacement checks |
| `test_utils.py` | General loading/conversion utilities, local Python model files, and Qwen3.5 patch-weight layouts |
| `test_generate.py` | Generation, sampling distributions, AR/server positioned samplers, stopping criteria, structured logits, and thinking-phase state |
| `test_extraction_models.py` | GLiNER candidate pools, span/schema handling, checkpoint loading, and privacy tagging/quantized inference |
| `test_pp_doclayout_v3.py` | Document-layout detection configs, sanitization, forward outputs, and postprocessing |
| `test_tokenizer_utils.py` | Streaming detokenizers, decoder detection, and tokenizer wrappers |
| `test_smoke.py` | Manual model-download runner, excluded from automated collection |

`test_speculative.py` contains **1,399 formatted lines**, down from 2,027 at
`a1bbdca6`. Reusable model constructors, checkpoint tensors, and transaction doubles
share `test_models.py`, and plain config/shape profiles share `model_cases.json`.
Together those files shrink by **35 lines**; the 628-line speculative-file
reduction includes relocation. Training imports its two config helpers directly
from `test_models.py`. There are no new fixture or test files.

All **395 speculative cases pass**, retaining the former 413 cases' scenarios:
dispatch assertions share the round matrix, and acceptance/budget/error scenarios
share one runner. Full suite: **1,773 passed, four skipped, 39 passing subtests**.
Both isolated (15,723 lines / 2,457 branches) and full-suite (80,997 / 12,836)
production execution sets are unchanged, with zero lost or added paths. See
[speculative_coverage.md](speculative_coverage.md) for exact size accounting,
validation scope, and the earlier intentional pruning. Current suite size is
**23,698 Python lines + 2,543 JSON lines = 26,241 combined**.

`test_cli.py` contains **498 formatted lines**, down from 938 at `9383ef6c`
(440 lines / 46.9% fewer), with **37 collected cases** before and after.
Use `_args` for fresh real-parser defaults and `_text_cli` for dispatch doubles.
Parameter tables cover SAM3 routing/thresholds, RF-DETR flags/annotations,
diffusion formatting/options, native video versus sampled frames, early modality
routing, and video reference/keyframe workflows. Helpers stay in this file.

Real parser checks replace the four AST flag checks, and the text CLI checks
actual system-prompt forwarding instead of looking for an assignment in source.
The immutable-default identity check and private visualizer patch flag are
removed; default-value parity and visualizer delegation remain asserted.
Subprocess shutdown cleanup, terminal redraw/restore, audio validation before
loading, optional-argument fallback, and unknown-frame-count progress remain.

The full-suite comparison against `9383ef6c` passes **1,791 tests, four skips and
39 subtests** on both revisions. Executed production lines increase from
**80,947 to 80,997**, and branch outcomes from **12,834 to 12,836**, with **zero
lost paths**. Additions come from importing/exercising the chat parser and actual
system-prompt dispatch. Black, isort, pyflakes, autoflake and whitespace checks
pass. These are measured Python execution sets, including imports; they exclude
subprocess execution, native kernels and skipped optional checks. No production
code or additional test files change.

The suite has 27 test modules. Attention checks share `test_attention.py`;
Qwen3.5 patch-weight sanitization lives with loading checks in `test_utils.py`.
Responses normalization and tool-stream finalization share `test_server.py`;
structured-output processors share `test_generate.py`. Vision LRU checks live in
`test_cache.py`, while all 45 cached-image source checks live in `test_models.py`.
Those source checks verify the keyword's presence, not runtime cache reuse.
GLiNER and privacy-filter checks share `test_extraction_models.py`.

The consolidation at `6ce9f04d` removed 427 Python lines and six test files
against `f7de22df`, retaining both JSON files. Its full-suite execution sets were identical:
80,634 production lines and 12,715 branch outcomes, with no lost or added paths.
That validation reported 1,852 passed, five skipped, and 41 passing subtests.
These are measured execution sets, not complete production coverage. The earlier
pruning tradeoffs still apply. The suite now contains 24,695 Python lines plus
2,229 readable JSON lines, or 26,924 combined.

After removing the unused cache helper and redundant standalone mask test, the
full suite reports 1,875 passed, five skipped, and 41 passing subtests. Coverage
against `66f67781` retains exactly 80,919 production lines and 12,819 branch
outcomes, with no lost or added paths.

Sampling checks share `test_generate.py`, which contains 1,877 lines after a
76-line reduction across generation and sampling tests. Parameterized runners
retain top-p shape/dtype checks, sampler survivor sets, invalid parameters,
post-error compiled sampling, and the shared AR/server positioned-sampler
contract. Independent NumPy references retain all 40 trials per filter and the
existing numerical boundary tolerances. The peaked p-less assertion shares its
reference test; invalid parameters and post-error sampling share one runner.
Removing the duplicate float32 filtered-shape and negative typical-p variants
reduces 20 sampling cases across seven functions to 17 across five, saving five
more lines against `c9b35c2d`. All 114 generation cases pass. Focused coverage
retains exactly 6,908 production lines and 717 branch outcomes, with no lost or
added paths. Two former subtests now run as parameterized cases.
The full suite reports 1,876 passed, five skipped, and 39 passing subtests;
its execution sets remain 80,919 production lines and 12,819 branch outcomes,
with no lost or added paths.

`test_tool_parsers.py` discovers modules under `mlx_vlm.tools.parsers` and imports
them through `load_tool_module`. Add a literal native-format example to
`WIRE_CALLS` for each new parser; the inventory check requires matching module,
registry, and example names. Shared checks cover argument values/types, single and
repeated extraction, unique IDs/indices, surrounding prose, template variants,
processor inference, overrides, and priority over the JSON fallback. Add wire
variants to `WIRE_VARIANTS`; the same runner checks Mistral v3/v11, bare EOF calls,
and single/repeated extraction with prose. Prose-only inputs, including text
containing `call:`, also run through every parser.

Add syntax variants to `test_parser_syntax` with the parser name, expected argument
type, input, expected call or list of calls, and optional tool schema. Gemma,
Pythonic, Cohere object/array payloads and escaping, and GLM newline handling share
this runner while retaining their return-type assertions. Error cases use
`test_invalid_calls`; no test function has a parser-specific implementation.
The 13 formats run 46 cases in 261 lines. Folding three cases into the common
runner reduces collection from 49 without dropping their assertions. Against
`3d5c0510`, isolated execution sets remain 4,068 production lines and 247 branch
outcomes; full-suite sets remain 80,919 lines and 12,819 branch outcomes, with no
lost or added paths. The full suite reports 1,876 passed, five skipped, and 41
passing subtests. Relative to `6ce9f04d`, parser coverage still adds 285 production
lines and 104 branch outcomes, with no lost paths.

MoE offload remains separate and shrinks from 615 to 305 lines. Reuse its
checkpoint/quantization/repacking and relative-parity helpers. Separate-projection
and fused-expert checkpoints share a parameterized loader check; direct patching,
threaded expert access, raw Mixtral-style sanitization, and malformed/disk-space
failure checks remain explicit. String and Path checkpoint destinations are both
exercised. All numerical tolerances and per-test hardware/dependency skips remain.

APC settings and worker shutdown checks live in `test_server.py`; APC builders
and parameterized cache-format checks live in `test_apc.py`. Quantized cache
lifecycle checks share `test_cache.py`, while TurboQuant numerical and kernel
checks remain in `test_turboquant.py`.

Audio model checks share tiny configuration defaults within `test_audio_models.py`.
Optional audio dependencies and real VoiceChat checkpoint skips remain scoped to
their individual tests. The speech-generation subprocess imports its tiny model
from this same module. MiniMax H3 workflows share conditioning, pipeline/request
construction, and canonical weight setup in `test_video_generation_models.py`;
their numerical references and cached/uncached equality checks remain explicit.

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
Thinking formats share `_THINKING_CASES` between endpoint and stream-state tests;
response-template tool checks retain both Chat streaming and Anthropic responses.
Keep parser extraction in `test_tool_parsers.py` and prompt-only image placement
in `test_prompt_utils.py`; Responses conversion and stream filtering stay here.
Realtime doubles expose the event fields consumed by the server without importing
a model-specific event class. See [server coverage measurements](server_coverage.md)
for the current line counts and exact coverage comparison.
The audio HTTP and realtime checks also live here. `audio_client` depends on
`reset_audio_runtime` to isolate model caches, metrics, and audio-queue shutdown;
`realtime_client` owns the voice engine lifecycle. These fixtures are scoped to
their tests and do not change the ordinary server client's setup.

The audio merge against `c6997db7` retains all 14 moved cases and all 208 combined
server cases, removing one file and 17 Python lines. Isolated coverage is unchanged:
10,563 production lines and 1,665 branch outcomes. Full-suite execution sets also
remain 80,919 lines and 12,819 branch outcomes, with no lost or added paths;
1,876 tests pass with the same five skips and 41 passing subtests.

`test_image_generation_models.py` owns the six image-generation families and
keeps checkpoint writers, pipeline doubles, and tiny quantization models in the
same file. `image_generation_cases.json` supplies ordinary data for six component
checks across four model families, eight generation/edit wrapper cases, and four
weight-key sanitizer cases. Each entry in `models` names its import `module`
relative to `mlx_vlm.models`, `model_class`, `config_class`, ordinary constructor
`config`, and `checks`. The runner imports the module dynamically and constructs
the named classes; `config_class: null` passes config fields directly to the model
constructor (Mage Flow). The case `id` only labels the collected test. For example,
Z-Image's transformer imports `z_image.transformer` and constructs
`ZImageTransformer(ZImageTransformerConfig(**config))`.

Requests, expected result fields, forwarded arguments, and metadata belong in
`wrappers`. Sanitizer keys specify their source shape and expected destination
(`null` means drop). Python constructs models, supplies tensor layouts and calling
conventions, and checks the results. No references or executable expressions are
needed in JSON.

Behavioral checks use `_ModelFamily` for dynamic submodule access, such as
`ernie.config` or `flux.weights`. These resolve to real imported modules, so
patches still apply to production objects without a static model import block.

The four `downloads` cases use one model-independent runner. Each supplies an
import module, download `config`, expected `repo_id`, and optional destination
or required-pattern expectations. The runner mocks Hub access and validation,
checks argument forwarding, destination creation, and validation calls, and
leaves required-file checks to the existing layout tests.

Reuse the parameterized discovery, dimension, download, conversion, and
weight-loading checks with descriptive family IDs. Retain family-specific prompt,
guidance, position, VAE, and numerical assertions next to the shared contracts,
including Z-Image padding and Ernie conditioning. CLI routing and request
forwarding for every output modality live in `test_cli.py`; image HTTP endpoints
stay in `test_server.py`. Audio transcription formats and translation share a
request runner in `test_server.py`. Loading tests share checkpoint patches
and model doubles in `test_utils.py`, with parameterized quantization policies.

## JSON model cases

`model_cases.json` contains 51 configurable contract cases and 17 native
forward/cache cases, plus five APC configuration profiles. `test_models.py`
collects each contract case separately with a stable model ID, so failures can
be selected with `pytest -k`:

```sh
python -m pytest -q mlx_vlm/tests/test_models.py -k llava_bunny
```

The `apc` section supplies ordinary settings to `test_apc.py`. Gemma4, Qwen3.5,
and Qwen4 Exp profiles name an existing `case` and override its nested `config`;
LFM2 and Z1T profiles provide a `module` and their complete input `config`.
`apc_config` deep-copies and recursively merges these settings before calling
the shared `build_config` helper, so derived layer layouts reflect the APC
settings and neither the base case nor the profile is mutated. The module is
imported dynamically under `mlx_vlm.models`. Model/component construction and
APC assertions remain in Python; these profiles do not add model-contract cases.

The five effective configs, including derived fields, match their previous
Python definitions exactly. The configuration migration removed 104 Python lines
and added 87 JSON lines, saving 17 combined lines.

The compact APC suite replaces the validated prototype in `test_apc.py` and runs
under default collection. It contains **1,499 lines instead of 3,187** and
**97 cases across 35 shared test functions**, with all helpers inline. Shared
runners cover 19 cache layouts, disk restoration, memory-growth matrices, prefix
generation, and packed quantization. Prototype files are removed. Coverage-based
pruning also removes the optional live checkpoint-backed smoke, whose skipped
coverage was not measured.

The full default suite reports **1,791 passed, four skipped, and 39 passing
subtests**. Against `eda5edef`, all 80,919 previously covered production statements
and 12,819 branch outcomes remain covered, with seven statements and three
branch outcomes added. See [APC coverage](apc_coverage.md) for exact measurements,
pruned checks, retained tolerances, and reproduction commands.

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

Gemma assistant mask checks share `test_drafter_masks` in `test_speculative.py`.
Its eight cases cover local rotating-cache offsets with real MLX masks, including
the former standalone assertion that position 128 clamps to a cache length of 8.

`test_smoke.py` is the manual model-download runner and is excluded from CI's
automated suite.
