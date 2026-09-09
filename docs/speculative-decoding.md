# GLM-5.3-Flash MTP decoding

The speculative implementation starts with one target, GLM-5.3-Flash, and
its native next-token prediction layer. Its FP8 checkpoint can be loaded
through the shared FP8-to-MXFP8 conversion and used with an extracted MXFP8
MTP head. Existing converted GLM-5.3-Flash MTP checkpoints remain loadable.
Other drafter families and their legacy verification/rollback APIs have been
removed in this rebuild.

## Ownership

| Component | Responsibility |
| --- | --- |
| `speculative/cache_state.py` | Request state, target/draft caches, positions, feature alignment, commit/abort |
| `models/cache.py` | KV storage and bounded recurrent/pooling histories |
| `speculative/mtp.py` | Propose, run the target, accept, emit |
| `speculative/sampling.py` | Model-independent prefix acceptance and target sampling |
| `speculative/drafters/glm5_next_mtp/` | Checkpoint extraction and stateless MTP forward |
| `speculative/utils.py` | CLI and server generation entry points |

The MTP model owns weights. It has no request cache, hidden-state seed,
position cursor, reset method, or rollback callback. Two simultaneous requests
can use the same model with independent `SpeculativeCache` objects.

## Round contract

1. Prefill the target and retain its final normalized hidden states across
   every prompt chunk. MTP consumes those features paired with tokens shifted
   one position to the left, ending with the first sampled target token.
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

No new model-specific Metal kernels are introduced. Acceptance is a small
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
  --batch-size 1 --block-sizes 2 3 5 --output results.json

python -m pytest mlx_vlm/tests/test_speculative.py \
  mlx_vlm/tests/test_speculative_transactions.py mlx_vlm/tests/test_cache.py
```

Block size includes the target bonus (`2` means one draft). A caller's explicit
block size is honored; there is no model-specific adaptive policy.

The initial scope includes greedy and temperature sampling, chunked prefill,
and left-padded/ragged batches. APC target-only snapshots do not contain the
MTP state, so prefix reuse is disabled for this path. Logits processors and
rotating/TurboQuant caches are rejected explicitly. Native KV and batch KV
caches, native quantized KV, recurrent arrays, and pooling caches participate
in transactions. Model weights may be FP8 independently of the KV format.

## FP8 validation

[Recorded short-run results](benchmarks/glm53-mtp-fp8.json) cover 12 MTP
comparisons against ordinary decoding: batch sizes 1 and 2, block sizes
2/3/5, and explanation, code, and counting prompts. Every generated token
matched its baseline. With one draft, measured throughput was 22.2–22.5
versus 17.8 tokens/s for batch one, and 31.0–33.0 versus 27.3–27.4 aggregate
tokens/s for batch two. A two-draft coding run reached 24.9 tokens/s; larger
blocks also slowed other prompts. These are short runs, not a general speedup
guarantee or a long-context benchmark.

## Upstream references

The organization follows the proposal/verification/acceptance split visible
in [vLLM's proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/llm_base_proposer.py)
and [SGLang's speculative state](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_info.py).
This implementation adapts that split to MLX's caches and lazy execution;
it does not copy CUDA scheduling or paged allocation.

See also [vLLM MTP](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/),
[SGLang speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding),
and the [official FP8 checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash).
