# Cache-owned MTP decoding

GLM-5.3-Flash FP8 with its native MTP head is the primary target. Qwen3.5
is a second adapter using the same generation loop and cache transaction
contract. Neither drafter owns request state. Legacy DFlash, EAGLE, DSpark,
and model-specific speculative verifiers are removed.

GLM's official FP8 checkpoint loads through the shared FP8-to-MXFP8 conversion;
the native MTP head can be extracted and quantized independently. Existing
converted GLM MTP checkpoints remain loadable. Qwen3.5 extraction handles
its native dense and MoE MTP weights; real-model validation uses Qwen3.5-0.8B.

## Ownership

| Component | Responsibility |
| --- | --- |
| `speculative/cache_state.py` | Request state, target/draft caches, positions, feature alignment, commit/abort |
| `models/cache.py` | KV storage and bounded recurrent/pooling histories |
| `speculative/mtp.py` | Propose, run the target, accept, emit |
| `speculative/sampling.py` | Model-independent prefix acceptance and target sampling |
| `speculative/drafters/` | Checkpoint extraction and stateless GLM/Qwen MTP forwards |
| `speculative/prefix_cache.py` | Atomic target/draft/seed checkpoints in the bounded APC store |
| `speculative/utils.py` | CLI and server generation entry points |

The MTP model owns weights. It has no request cache, hidden-state seed,
position cursor, reset method, or rollback callback. Two simultaneous requests
can use the same model with independent `SpeculativeCache` objects.

## Round contract

1. Stream each target prefill chunk into MTP, pairing its final normalized
   features with tokens shifted one position to the left. The final chunk ends
   with the first sampled target token. Retain only the next draft prediction
   and its hidden state; the full prompt feature tensor is not retained.
2. The cache stores the next MTP prediction and its hidden state. Reuse that
   seed and run additional MTP steps to propose a fixed number of tokens.
3. Verify `[previous target token, drafts...]` with ordinary target forwards
   inside a bounded cache transaction.
4. Accept the matching draft prefix and emit one target correction or bonus.
5. Retain exactly as many target input positions as were emitted. Each
   recurrent/pooling cache restores its own accepted state; KV caches trim
   their unaccepted tails. No rollback replays the target.
6. Discard hypothetical draft expansions. Extend the MTP cache with committed
   output tokens paired with the verified *target* features, then retain only
   the last prediction and hidden state for the next round.

Both target and draft caches end a round at the same logical position. The
last emitted token remains pending for the target's next forward, matching
ordinary autoregressive generation. Ragged batches commit per-row lengths;
finished rows retain no further input positions. The last output of a round
is yielded only after commit. Closing a generator partway through a round
commits only the prefix actually delivered. Failures abort open transactions.

Drafts use argmax. At nonzero temperature, a target draw is accepted when it
matches that deterministic proposal. Otherwise the draw is the correction.
This is rejection sampling with a point-mass proposal distribution; it needs
no draft RNG or probability buffer. Positioned target samplers retain their
per-request generated-token positions.

## Shared operations

Positioned samplers draw a complete verification block in one batch when there
are no history-dependent processors. This preserves each row/position RNG key
and transfers one prefix decision to the host. History-dependent sampling
continues along accepted prefixes only.

The shared short-block projection helper uses the existing dense GEMV kernel
for singleton BF16/FP16 rows, including tied embedding output heads. MXFP8 MoE
gate/up projections fuse into one dispatch and reuse selected expert weights
across adjacent tokens. The dispatch is based on tensor shape and quantization
format, with native fallbacks for other shapes and formats; it has no model-name
or speculative-engine flags. Expert addresses use wide byte offsets.

No model-specific Metal kernels are introduced. Acceptance is a small
compiled MLX operation. Attention, gated-delta history, hyperconnections, and
quantized projections use shared operators in `models/`. Shared operators
are available to ordinary decoding and future model adapters. The Qwen batch
forward used by ordinary AR generation is retained as `batch_invariant.py`;
it is no longer a speculative verifier or a source of rollback callbacks.

A future model adapter must return the required target features and implement
an ordinary draft forward. Its cache types must implement the same transaction
contract. It must not add model-name dispatch or rollback hooks to the loop.

## Run and validate

```sh
python -m mlx_vlm.split_mtp --model zai-org/GLM-5.3-Flash \
  --output GLM-5.3-Flash-MTP-FP8 --q-mode mxfp8

mlx_vlm.generate --model zai-org/GLM-5.3-Flash \
  --draft-model GLM-5.3-Flash-MTP-FP8 --draft-kind mtp \
  --draft-block-size 2 --temperature 0 --prompt "Explain why the sky is blue."

PYTHONPATH=. python examples/benchmark_glm53_mtp.py \
  --model /path/to/GLM-5.3-Flash-FP8 \
  --draft-model /path/to/GLM-5.3-Flash-MTP-FP8 \
  --batch-size 1 2 --context-tokens 0 1024 --temperatures 0 0.8 \
  --max-tokens 256 --block-sizes 2 3 --output results.json

python -m pytest mlx_vlm/tests/test_speculative.py \
  mlx_vlm/tests/test_speculative_transactions.py \
  mlx_vlm/tests/test_speculative_serving.py mlx_vlm/tests/test_cache.py
```

Block size includes the target bonus (`2` means one draft). A caller's explicit
block size is honored; there is no model-specific adaptive policy.

Greedy and temperature sampling, chunked prefill, left-padded/ragged batches,
logit bias, repetition/presence/frequency penalties, and other history-based
logits processors use this path. Processors see the full prompt plus only
committed output tokens, including after prefix reuse. Processors requiring
external updates after each token (such as structured output) yield one target
token at a time; they preserve correctness but do not gain speculative speedup.
Thinking budgets mask the target distribution at each sampled position, forcing
the configured closing sequence once the reasoning budget is exhausted. Zero
budgets, multi-token delimiters, natural termination, and boundaries inside a
speculative block use the same policy as ordinary AR. Only committed history
and the candidate accepted prefix determine the mask; rejected tokens cannot
advance a reasoning counter. Rotating/TurboQuant caches remain unsupported.
Native KV, batch KV, native quantized KV, recurrent arrays, and pooling caches
participate in transactions. Model weights may be FP8 independently of KV format.

Request metrics also belong to the cache, with separate accepted/drafted/round
counters for every row. A prefix hit starts new counters. Completion responses
report that row's counters, so overlapping requests never share metrics through
the draft model.

Pass `logprobs=True` to generation or request `logprobs` from the server to
return verified target probabilities, including the first token, rejected-draft
corrections, and bonus tokens. Server top logprobs are available within its
configured cap. Values use processed target logits before temperature/top-P,
matching ordinary AR. Forced reasoning tokens have probability one under the
constrained distribution. Speculative calls without this option return no
probability payload.

## Prefix reuse

A checkpoint contains target caches, draft caches, logical position, the pending
target token, positional offsets, and the next MTP prediction/hidden state. It is stored through the
existing APC budget, LRU, cloning, and disk serialization using native cache types.
Keys are separated by target/draft checkpoint identity, schema, and the existing
request/tenant/media salt. A target-only checkpoint cannot satisfy an MTP lookup.

If the target has processed N inputs, the MTP cache has already consumed the
shifted token at N. The prefix key must therefore include N+1 tokens. A hit
restores both caches and processes the last key token as the first target suffix
input. This preserves MTP alignment on identical prompts and conversation
extensions. Changing that pending token prevents reuse of that checkpoint.

Prompt chunk boundaries and completed generations can be checkpointed. Closing
a stream first commits only delivered tokens, then stores that committed state.
`PromptCacheState` uses the same bounded checkpoint mechanism across turns.
The server prefills restored suffixes independently when prefix lengths differ,
then merges their target/draft caches for one batched decode. Cold-only requests
retain batched prefill. This avoids target replay, with an added prefill scheduling
cost for mixed warm/cold admissions.

Qwen's ordinary forward uses the existing shared short-block projection helpers
and its existing convolution/attention operations with single-token reductions.
Active cache transactions preserve cache object identity. Batch cache padding
changes replace their metadata arrays, invalidating cached attention metadata.
No new Metal kernel is introduced for the second adapter.

## Validation and performance

The latest [controlled optimization comparison](benchmarks/glm53-mtp-fp8-optimized.json)
uses an M3 Ultra with 512 GB RAM and MLX 0.32.2. It measures 256 output tokens
per row, one draft plus bonus, and two alternating before/after repetitions
with the same loaded FP8 target and MTP weights. Every output matches AR:

| Batch | Temperature | AR tok/s | Previous MTP tok/s | Optimized MTP tok/s | MTP improvement | Speedup over AR |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 18.25 | 25.03 | 30.60 | 22.3% | 1.68× |
| 1 | 0.8 | 17.99 | 23.01 | 28.29 | 22.9% | 1.57× |
| 2 | 0 | 27.66 | 34.10 | 34.80 | 2.1% | 1.26× |
| 2 | 0.8 | 27.55 | 29.99 | 31.18 | 4.0% | 1.13× |

Rates for batch two are aggregate. The comparison substitutes the projection
and acceptance functions from `378349e9` into the current cache-owned request
loop, then enables the optimized functions. Target and MTP prefill are measured
separately; decode rates count the remaining 255 tokens per row. These runs use
raw encoded prompts without a chat template: a Fibonacci coding prompt for batch
one, plus a sky explanation for batch two. Settings, exact prompts, tokens, and
per-run timings are in the artifact. This short-context comparison does not
establish the same speedup for long contexts or constrained sampling.

The [FP8 reporting checks](benchmarks/glm53-mtp-fp8-reporting.json) compare 64
sampled output tokens and their full target logprob vectors against AR, with no
thinking limit and with budgets of 0 and 8. All three token sequences and every
logprob match exactly. Forced closing tokens have a reported logprob of zero.
The checks use the model's chat template with thinking enabled.

The latest [Qwen3.5-0.8B matrix](benchmarks/qwen35-mtp-optimized.json) also passes
all 12 token comparisons. One draft reaches 144.1–230.3 tok/s versus
127.0–137.3 tok/s for AR at batch one. At batch two it reaches 205.7–217.2
aggregate tok/s versus 305.4–334.2 for AR. Batch-two MTP remains slower for this
small model despite the shared optimizations.

Earlier measurements below establish longer-output, prefix-cache, and
multi-context coverage before these projection and sampling optimizations.
The primary measurements use GLM-5.3-Flash FP8 target weights and an MXFP8
native MTP head on MLX 0.32.2. All recorded comparisons match every ordinary
AR token:

| Workload | AR | One draft + bonus | Speedup |
| --- | --- | --- | --- |
| 256 outputs, batch 1, 22–1,048 prompt tokens, T=0/0.8 | 17.85–18.14 tok/s | 19.60–24.53 tok/s | 1.09–1.36× |
| 256 outputs per row, batch 2, same prompt range and temperatures | 27.44–27.68 aggregate tok/s | 28.82–30.92 aggregate tok/s | 1.05–1.12× |
| 1,024 outputs, 4,123-token prompt, T=0 | 17.55 tok/s | 22.53 tok/s | 1.28× |
| 1,024 outputs, 4,123-token prompt, T=0.8 | 17.22 tok/s | 21.70 tok/s | 1.26× |

The [256-token matrix](benchmarks/glm53-mtp-fp8-256.json) has 24 comparisons
across two prompts, batch sizes 1/2, temperatures 0/0.8, and blocks 2/3.
The [longer runs](benchmarks/glm53-mtp-fp8-1024.json) reach about 343.2 GB peak
MLX memory. The 256-token matrix includes initial MTP alignment in decode time;
the later harness accounts for both target and MTP prefill in `prefill_seconds`.
All throughput runs use a fixed output budget and continue past EOS, to keep
workload lengths comparable. Larger blocks sometimes lose throughput.

The [phase profile](benchmarks/glm53-mtp-fp8-profile.json), on a 256-token coding
prompt, attributes roughly 91–92% of round time to target verification. With one
draft, target verification averages 74.3 ms, acceptance 0.8 ms, and commit/MTP
alignment 5.6 ms. With two drafts, target verification averages 102.6 ms and
alignment 6.2 ms. Phase boundaries explicitly synchronize MLX, so these timings
are diagnostic and should not be used as normal throughput measurements. Use
`--profile` to reproduce them; normal decoding has no phase synchronization.

The [FP8 serving check](benchmarks/glm53-mtp-fp8-serving.json) compares AR, cold
MTP, and warm MTP with temperature 0.8 and repetition/presence/frequency penalties.
All 128 outputs match; the warm request reuses 148 of 149 prompt tokens. Its
measured prompt-processing time falls from about 1.40 s cold to 0.057 s warm.

[Qwen3.5-0.8B results](benchmarks/qwen35-mtp.json) cover 12 comparisons of 128
outputs, batch sizes 1/2, temperatures 0/0.8, and blocks 2/3. Every token matches.
One draft reaches 106.5–158.0 tok/s versus 103.6–106.4 for batch one. Batch two
is slower with MTP (151.1–200.1 versus 294.0–320.9 aggregate tok/s). This adapter
validates the shared cache contract; it is not a recommendation to enable MTP
for every Qwen workload. Dense and MoE variants also share the serving unit tests.

Tests cover warm/cold batches, disk checkpoint restore, changed pending tokens,
processor histories, external per-token processor updates, cancellation, and
per-request positional state. Native caches reject model forwards that replace
cache objects during an active transaction. These tests supplement the original
[short FP8 runs](benchmarks/glm53-mtp-fp8.json).

## Upstream references

The organization follows the proposal/verification/acceptance split visible
in [vLLM's proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py)
and [SGLang's speculative state](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_info.py).
This implementation adapts that split to MLX's caches and lazy execution;
it does not copy CUDA scheduling or paged allocation.

See also [vLLM MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/),
[SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding),
and the [official FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash).

Thinking-budget behavior follows sampling-time constraints in
[vLLM's thinking-budget state](https://github.com/vllm-project/vllm/blob/9d3e991ece7f54c173c0cc9261ab5969b1dbeb4c/vllm/v1/sample/thinking_budget_state.py)
and [SGLang's reasoning grammar](https://github.com/sgl-project/sglang/blob/ae1acf822dd357641d885f30c58d2ad547ec9220/python/sglang/srt/constrained/reasoner_grammar_backend.py).
The MLX policy reads the cache-owned token prefix directly instead of maintaining
a second speculative reasoning cursor that needs rollback.
