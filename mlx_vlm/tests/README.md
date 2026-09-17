# Test organization

The suite has **15 test modules plus `conftest.py`: 16 Python files**.
`test_extraction_models.py` remains independent. The manual `test_smoke.py`
download runner has been deleted.

Add tests and reusable helpers to the module covering the same production
behavior. Import shared helpers from that module instead of adding separate
fixture files. Keep dependency, hardware, and checkpoint skips scoped to the
tests that need them.

Run the automated suite from the repository root:

```sh
python -m pytest -q mlx_vlm/tests
```

Use a module path or `-k` to select a smaller group while developing.

| Test module | Scope |
| --- | --- |
| `test_models.py` + `model_cases.json` | Shared language, vision, audio, projector, embedding, position and native forward/cache contracts; checkpoint loading, sanitization, and document layout |
| `test_model_ops.py` | Attention kernels and numerical parity, rotary embeddings, weight quantization, packed Hadamard checkpoints, and format conversion |
| `test_cache.py` | Cache lifecycle, APC lookup and prefix reuse, adapters, memory budgets, disk persistence, TurboQuant, batched masks/attention, and vision-feature LRU behavior |
| `test_processors.py` + `processor_cases.json` | Tokenizers and detokenizers, processor loading and media contracts, image/video utilities, prompt construction, and dynamically discovered tool parsers |
| `test_audio_models.py` | Audio/omni model components, speech generation, streaming, checkpoint conversion, audio loading, downmixing, and resampling |
| `test_video_generation_models.py` | Video model components, conditioning workflows, cached trajectories, numerical references, conversion, discovery, request/result adapters, and audio/video muxing |
| `test_extraction_models.py` | GLiNER candidate pools, span/schema handling, privacy tagging/quantized inference, and Sapiens2 vision extraction |
| `test_image_generation_models.py` + `image_generation_cases.json` | Bonsai, Flux2, Ideogram4, Z-Image, ERNIE Image, and Mage Flow generation/editing, components, loading, and conversion |
| `test_diffusion_models.py` | LLaDA, Nemotron, and DiffusionGemma models/generation, numerical parity, caches, vision, sanitization, and generation-config loading |
| `test_speculative.py` | Drafter loading and compatibility, generation parity, verification, cache transactions, and quantized speculative state |
| `test_generate.py` | Generation, sampling, stopping criteria, EOS reset behavior, structured logits, and thinking-phase state |
| `test_server.py` | Chat/Responses/Anthropic APIs, image endpoints, batching/cancellation, runtime settings, reranking, tool stream state, HTTP audio, and realtime voice sessions |
| `test_cli.py` | Text/image/audio/video CLI routing, arguments, diffusion display/visualizers, detector display options, and CLI/library default parity |
| `test_trainer.py` | Training workflows, adapter loading, MRoPE/gated-delta gradients, and MoE gradient/expert-replacement checks |
| `test_moe_offload.py` | MoE checkpoint repacking, expert offload, output parity, and failure handling |

The former `test_utils.py` cases follow ownership: model/checkpoint loading in
`test_models.py`, media handling in `test_processors.py`, weight conversion in
`test_model_ops.py`, EOS handling in `test_generate.py`, adapter delegation in
`test_trainer.py`, and diffusion configuration in `test_diffusion_models.py`.
Speculative setup stays in `test_speculative.py`; only general JSON config
construction is shared through `test_models.py`.

## Current validation

The merge of `main` at `10db0927` preserves the consolidated **16-file layout**,
JSON runners, and prior deletions. New upstream tests join the existing suites:
APC restore accounting in `test_cache.py`, Qwen Omni DeepStack in
`test_audio_models.py`, Sapiens2 in `test_extraction_models.py`, packed Hadamard
checkpoints in `test_model_ops.py`, GLM head sanitization in `test_models.py`, and
periodic cache evaluation in `test_generate.py`. The shared Anthropic runner now
expects upstream's empty-string assistant content for tool calls. No existing
test function was removed by this merge.

The full suite passes **1,807 tests, four skips, and 41 subtests**. It contains
**22,873 Python lines plus 3,256 JSON lines: 26,129 combined**. Formatting, lint,
Python 3.10 syntax, JSON parsing, and whitespace checks pass.

Production code matches `main` at `10db0927`, so both coverage runs use the same
**153,776 statements and 40,444 branch outcomes**. The merged PR executes
**82,142 statements (53.4167%) and 13,175 branch outcomes (32.5759%)**;
`main` executes **102,954 statements (66.9506%) and 18,311 branch outcomes
(45.2749%)**. Main passes 4,287 tests with nine skips and 162 subtests in the same
environment. Both runs disable downloads and omit test code from coverage;
the manual smoke runner is excluded on main. Earlier intentional pruning remains.

## Previous image-contract validation

The image-contract refactor against `31218fb1` keeps all existing scenarios and
uses seven named runners for **43 JSON cases**: forward (6), wrapper (8), sanitize
(9), download (4), layout (6), quantized load (8), and save/reload (2). Dispatch
and invalid-dimension checks use the same six-family registry. Numerical and
stateful pipeline checks remain in Python, alongside shared checkpoint builders.

| Image test source | Before | After | Change |
| --- | ---: | ---: | ---: |
| Python | 1,780 | 1,611 | -169 |
| Readable JSON | 249 | 406 | +157 |
| Combined | 2,029 | 2,017 | -12 |

The current suite totals **21,396 Python lines and 3,256 JSON lines: 24,652
combined**, with the same 16 Python files. Generalization saves considerably less
combined source than Python alone; configuration moved to JSON is counted above.
Sanitizer checks now compare nonzero tensor values as well as keys and shapes.

The image suite passes **140 cases** (previously 139: source/native Z-Image VAE
sanitization now collects separately). Its exact production execution sets remain
**8,034 lines and 682 branch outcomes**. The full suite passes **1,720 tests,
four existing skips, and 31 subtests**: 1,724 collected cases. Full-suite coverage
remains **80,981 executed lines and 12,824 branch outcomes**, with **zero lost or
added paths** against `31218fb1`. This preserves the immediate baseline; it does
not restore coverage intentionally pruned earlier in this branch.

Black, isort with the Black profile, autoflake, pyflakes, Python 3.10 syntax parsing,
JSON parsing, and whitespace checks pass. This pass changes tests and documentation
only. Use the coverage command below and compare against `31218fb1` to reproduce
the image-contract measurements.

## Previous compression and intentional pruning

Compared with `e48acdc6`, shared setup and runners reduce Python source
**22,331 → 21,565 lines**, saving **766 formatted lines**. JSON is unchanged at
**3,099 lines**; the combined total is **24,664 lines**, with the same 16 Python
files. All changes in this pass are tests or their documentation.

| Module | Before | After | Saved |
| --- | ---: | ---: | ---: |
| `test_generate.py` | 1,928 | 1,716 | 212 |
| `test_processors.py` | 2,506 | 2,317 | 189 |
| `test_diffusion_models.py` | 1,287 | 1,131 | 156 |
| `test_models.py` | 1,071 | 965 | 106 |
| `test_audio_models.py` | 1,613 | 1,573 | 40 |
| `test_video_generation_models.py` | 1,544 | 1,522 | 22 |
| `test_model_ops.py` | 1,021 | 1,002 | 19 |
| `test_cache.py` | 2,844 | 2,826 | 18 |
| `test_image_generation_models.py` | 1,784 | 1,780 | 4 |

Batch generation, prompt/media handling, diffusion dispatch, quantized cache
policies, and one-bit prompt/decode comparisons share setup. Synthetic checkpoint
weights and tiny configurations share builders; numeric references, seeds,
tolerances, masks, stateful cache checks, gradients, and concurrency checks remain.
Processor cases explicitly retain both fresh and initialized tokenizer padding
states when splitting the former sequential test into independent cases.

Remove **53 selected cases**: 45 cached-image source-substring checks, four
standalone dataclass default checks, two public annotation checks, one
historical video-default equality check, and the DeepSeek-V4 quantization-alias
regression. The alias regression was explicitly removed after the earlier
coverage-preserving compression; the production alias implementation is unchanged.
Parameterization exposes five previously in-function scenarios as separate
collected cases: **1,771 → 1,723 collected**. The full suite passes **1,719 tests,
four existing skips, and 31 subtests**.

Production coverage is **80,981 executed lines and 12,824 branch outcomes**.
Compared with `a5574e2a` (and `e48acdc6`), removing the alias regression loses
**18 lines and 12 branch outcomes**, with no added paths. This intentional
loss covers legacy DeepSeek alias mapping and loader selection of per-module
quantization overrides, including the explicitly disabled head. The preceding
compression preserved all 80,999 lines and 12,836 branch outcomes; other earlier
intentional pruning is documented below.

The preceding file consolidation against `acd30322` reduced **28 → 16 Python
files** and **22,608 → 22,331 Python lines**. Its 277-line saving comprised 263
lines from deleting the manual smoke runner and 14 from imports/module boilerplate.
That step preserved all 1,771 collected cases, 80,999 executed lines, and
12,836 branch outcomes.

Black, isort, autoflake, pyflakes, Python 3.10 syntax parsing and whitespace checks
pass. Environment: macOS arm64/Metal, Python 3.12.14, MLX 0.32.2,
Transformers 5.17.0, pytest 9.1.1 and coverage 7.16.1. Reproduce full-suite coverage:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
TOKENIZERS_PARALLELISM=false \
python -m coverage run --branch --source=mlx_vlm --omit='mlx_vlm/tests/*' \
  -m pytest -q mlx_vlm/tests \
  -p pytest_timeout --timeout=90 --timeout-method=signal -ra
python -m coverage json -o current-coverage.json
```

Compare exact `(file, line)` and `(file, branch-start, branch-end)` sets; additions
do not cancel losses. These measurements exclude tests, native kernels, child
process execution and optional skipped checks. The immediate baseline is `e48acdc6`. For the older `acd30322` baseline,
add `--ignore=mlx_vlm/tests/test_smoke.py`.

## JSON model cases

`model_cases.json` contains 51 configurable contract cases and 17 native
forward/cache cases, plus five APC configuration profiles. `test_models.py`
collects each contract case separately with a stable model ID, so failures can
be selected with `pytest -k`:

```sh
python -m pytest -q mlx_vlm/tests/test_models.py -k llava_bunny
```

The `apc` section supplies ordinary settings to `test_cache.py`. Gemma4, Qwen3.5,
and Qwen4 Exp profiles name an existing `case` and override its nested `config`;
LFM2 and Z1T profiles provide a `module` and their complete input `config`.
`apc_config` deep-copies and recursively merges these settings before calling
the shared `build_config` helper, so derived layer layouts reflect the APC
settings and neither the base case nor the profile is mutated. The module is
imported dynamically under `mlx_vlm.models`. Model/component construction and
APC assertions remain in Python; these profiles do not add model-contract cases.

APC tests and their helpers now share `test_cache.py` with the cache lifecycle
and TurboQuant checks.

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

## Earlier refactor notes

The following notes retain their historical module names, line counts, and
validation totals. Use the ownership table above for the current layout.

`test_processors.py` contains **997 formatted lines**, down from 1,898 at
`bfa15bb9`. An explicit JSON registry imports processors dynamically, shared family
defaults build fresh smoke-test inputs, and one checkpoint runner verifies native
routing, component types, configuration precedence and model attributes. The
12 checkpoint cases preserve direct/automatic loading and the missing-template
fallback. JSON contains configuration, shapes and expected values; calls and
assertions remain in Python.

Nine tokenizer loading/conversion cases and shared tokenizer doubles now live in
`test_tokenizer_utils.py`. Processor media contracts retain Qwen token order,
Gemma patch/audio masks, LFM resampling, Mage timestamps and embedding isolation,
Kimi literal control tokens, and DiffusionGemma cross-thread materialization.
Eleven formerly separate cases share related tests; their scenarios remain.
There are no separate fixture modules or production changes.

The file-size reduction includes relocation: **901 fewer processor lines**, offset
by **346 additional tokenizer-test lines and 529 JSON lines**. The three files
therefore go from **2,077 to 2,051 lines: 26 fewer combined**. Processor tests pass
**117 cases with one optional PyTorch skip**; the tokenizer suite passes **22**.
Full suite: **1,762 passed, four skipped, 39 passing subtests**. Compared with
`bfa15bb9`, all **80,997 executed production lines and 12,836 branch outcomes**
remain covered, with two added lines (Laguna's real chat-template property and
Muse's public cleanup method). Coverage excludes tests, native kernels, skipped
optional checks and subprocess execution. Black, isort, autoflake, pyflakes and
whitespace checks pass. At that revision, suite size was **22,608 Python + 3,099 JSON =
25,707 lines**, across 27 test modules.

The model/training Python refactor against `9d469e0f` saves **503 formatted lines**
without adding JSON, moving code to another module or changing production:

| File | Before | After | Saved |
| --- | ---: | ---: | ---: |
| `test_models.py` | 662 | 452 | 210 |
| `test_trainer.py` | 716 | 423 | 293 |
| **Combined Python** | **1,378** | **875** | **503** |

Model checks use plain pytest assertions and methods named after the JSON checks,
removing the duplicate method-name mapping. One position recorder handles chunked
prefill, cache-index lookup and request-owned RoPE deltas. Shared component setup
retains the same language/vision dtypes, projector dimensions, audio masks and
shape assertions. That revision retained all 45 cached-image source checks in the same order;
the current compression removes these declaration checks as described above.

Training shares dataset setup, batch collation, optimizer/save cases, adapter
construction and native/legacy adapter-loading contracts. Constructor assertions
run in the dataset fixture. Eight rotary gradient scenarios now collect as named
pytest cases using a shared native-versus-pure-MLX comparison, retaining both
input gradients, deterministic tensors, finite checks and 1e-4 tolerances. Gated
delta and MoE gradient/expert-replacement checks remain. The trainer changes from
22 tests plus eight subtests to **27 tests**: two rotary tests become eight cases,
and the standalone constructor case moves into the shared fixture.

Focused model/training/speculative validation passes **535 tests**. The full suite
passes **1,767 tests, four existing skips and 31 subtests**, versus 1,762 tests,
four skips and 39 subtests before. Exact production execution sets are identical:

| Scope | Executed lines before/after | Branch outcomes before/after | Lost / added |
| --- | ---: | ---: | ---: |
| Model + training + speculative suites | 39,639 | 4,343 | 0 / 0 |
| Full default suite | 80,999 | 12,836 | 0 / 0 |

Black, isort, autoflake, pyflakes, Python 3.10 syntax parsing and whitespace checks
pass. These offline Python coverage measurements exclude tests, native kernels,
subprocess execution and optional skipped checks; earlier pruning remains.

`test_speculative.py` remains **1,635 lines** and owns its model/drafter,
checkpoint and transaction setup. `test_models.py` exposes `build_config` and
`tiny_config(family, profile=None, **overrides)` for fresh JSON-backed configs.
Module names, config classes and named profiles live under `shared_configs` in
`model_cases.json`; `tiny_defaults` supplies common values. Training/checkpoint
settings stay distinct from language/inference variants, and explicit overrides
apply last. The preceding factory-only refactor saved four combined Python/JSON
lines.

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
within the suite. `test_models.tiny_config` supplies fresh JSON-backed configs
for speculative and training tests. `test_speculative.py` also owns MiniMax
speculative rollback checks; unrelated model tests remain in their existing
modules. Cache/position, sampling-parity, and batched-mask checks run through
shared contract runners.

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
a model-specific event class.
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
same file. `image_generation_cases.json` has a `families` registry for class
prefixes, pipeline classes, dispatch aliases, and invalid dimensions, plus a
`checks` object with seven groups: `forward`, `wrapper`, `sanitize`, `download`,
`layout`, `quantized_load`, and `save_reload`. `test_image_contract` dispatches
each entry to the corresponding `ImageChecks` method. IDs include both the
contract and case, for example `sanitize-z-vae-source`.

Each `forward` case names its import `module` relative to `mlx_vlm.models`, tiny
constructor `config`, input adapter, tensor shapes or token values, and expected
output shape. Class names follow the family prefix and component, with the config
class defaulting to the model class plus `Config`. For example, the
`z_image.transformer` entry constructs
`ZImageTransformer(ZImageTransformerConfig(**config))`. `config_class: null`
passes config fields directly to the model constructor (Mage Flow).

Wrapper cases supply requests, expected result fields, forwarded arguments, and
metadata. Sanitizer keys specify source shape, expected destination (omitted
means unchanged; `null` means drop), and optional transpose axes. The shared
runner compares all output keys and tensor values, including native-layout
roundtrips. Layout cases describe valid checkpoints or expected missing files;
quantization cases supply mode, bits, group size, and expected saved metadata.
Python constructs models and tensors, handles calling conventions, and asserts
results. JSON contains no references or executable expressions.

Behavioral checks use `_ModelFamily` for dynamic submodule access, such as
`ernie.config` or `flux.weights`. These resolve to real imported modules, so
patches still apply to production objects without a static model import block.

The four `download` cases use one model-independent runner. Each supplies an
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
