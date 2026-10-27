# Implementing speculative decoding

Speculative decoding is a lossless decode optimization. A small drafter proposes
several tokens, the target verifies the whole block, and MLX-VLM accepts the
matching prefix plus one target token. Rejected cache entries are then removed
before the next round.

```text
target prefill and hidden capture
              ↓
draft block → target block verification → accept prefix + target token
     ↑                                      ↓
     └──────────── cache rollback ──────────┘
```

## Code ownership

| Area | Responsibility |
|---|---|
| `generate/ar.py` | Prefill, target cache creation, and speculative dispatch |
| `speculative/drafters/` | Checkpoint loading, drafter architecture, and target compatibility |
| `speculative/dflash.py` | DFlash, DFlash2, and DSpark round loops |
| `speculative/mtp.py` | Native and assistant MTP round loops |
| `speculative/eagle3.py` | EAGLE-3 round loops |
| `speculative/common.py` | Ordinary verification forwards, acceptance, sampling state, and batch safeguards |
| `models/<family>/language.py` | Normal model traversal and hidden-state capture |
| `models/cache.py` | Bounded temporal state retention and accepted-state selection |
| `models/linear.py`, `models/switch_layers.py`, `models/fast_ops.py` | Shared decode-equivalent operations and fused kernels |
| `speculative/ops/` | Existing Qwen verifier operations |
| `speculative/cache_state.py` | Commit and abort of speculative cache transactions |

`draft_kind` selects the round loop, not the checkpoint architecture. Drafter
`model_type` values are mapped to `dflash`, `mtp`, or `eagle3` in
`speculative/drafters/__init__.py`.

## Adding a model

1. Inspect the real `config.json` and tensor names. Define the target layers,
   hidden-state inputs, block-size meaning, cache ownership, and quantization
   before implementing the drafter.
2. Add or reuse a drafter under `speculative/drafters/<family>/`. Implement its
   config normalization, checkpoint sanitization, `draft_block`, cache reset,
   and target compatibility checks.
3. Register its `model_type` with the correct `draft_kind`. Prefer architecture
   and config fields over repository-name checks.
4. Make target prefill return the hidden states the drafter consumes. DFlash and
   EAGLE-3 normally capture configured layers; MTP normally consumes final
   hidden state and may share target K/V.
5. Use the model's ordinary forward with a shared cache transaction. Stateful
   operators write through the temporal cache interface so rollback works
   without a second implementation of the layer. Put numerical dispatch in
   ordinary shared operators, using the same weights and module instances.
   Keep acceptance decisions and history-retention policy out of model layers.
6. Add synthetic contract tests, then validate the real target and drafter
   checkpoints before reporting support.

GLM and DeepSeek use `verify_forward` in the shared runtime. It starts one
cache transaction, calls the normal model, and returns the output and
transaction. MTP requests final hidden states; DFlash requests its configured
capture layers. The loop owns sampling, commit, and abort. Neither model needs
a speculative verifier or rollback method.

Normal forward calls expose hidden states through `return_hidden` or
`capture_layer_ids`. An ordinary `logits_from_hidden` method is useful when the
captured state needs architecture-specific normalization before the LM head.
Drafters own reshaping their inputs.

Existing model hooks remain supported for other architectures during migration;
new models should use the ordinary forward/cache contract. There is no target
registry, copied model view, or parameter-wrapper layer.

## Exact verification

A normal multi-token target forward is not automatically equivalent to repeated
one-token decoding. Kernel dispatch can change floating-point order, while
Mamba, gated-delta, convolution, or rotating caches need an explicit state for
the accepted position. Near an argmax tie, a small numerical change can alter
the generated sequence.

Use repeated ordinary one-token forwards as the reference. Short-block linear,
expert, and hyperconnection operators preserve that reduction order. Causal
attention preserves per-position cache ordering; image-prefill masks retain
their full visibility rules. The runtime splits larger verification blocks at
`DECODE_BLOCK_SIZE` while keeping one transaction across all parts. Large
ordinary prefill calls retain their bulk execution path. Verification must:

- produce the same greedy target tokens as autoregressive decoding;
- advance every cache through the verification block;
- restore stateful caches at `accepted + 1` tokens;
- handle zero, partial, and full acceptance;
- either support per-row batch rollback or require uniform acceptance;
- preserve the full required prompt hidden state through chunked prefill; and
- match every supported weight format, including quantized output heads.

Fused argmax and custom Metal kernels are optimizations, not correctness
shortcuts. Keep the full-logit fallback until parity is proven.

## Temporal caches for linear attention

`ArraysCache` owns a bounded history for the active verification window. Normal
decoding retains only the latest state. The same forward call can run inside a
cache transaction: no model-specific checkpoint or recurrent-layer replacement
is needed.

The shared gated-delta operators accept `cache` and a `cache_index`. They delegate
to `cache.update_recurrent`, which requests intermediate states from the kernel
only while history is needed. `cache.update_window` stores convolution or token
windows and retains their temporal views. Both methods support one block update
or several smaller updates within the fixed capacity.

Commit selects each row's accepted state; abort restores the starting state.
Both release the history. Over-capacity updates and incomplete histories are
rejected. A new recurrent operator needs to support the shared state-production
contract once; models using it do not need to know about MTP or draft lengths.
Numerical projection and attention dispatch is part of the ordinary operators,
independent of cache history retention.

## Validation and maintenance

Start with cheap tests for config normalization, strict weight loading, capture
layer order, block sizing, sampling state, and cache rollback. Compare cache
state and the next target token after zero, partial, and full acceptance for
single and batched generation.

Then run real-checkpoint greedy generation against the same autoregressive
baseline. Require token-for-token equality across several prompts and context
lengths before benchmarking. Measure all variants against one shared baseline
in the same process, report median decode throughput and acceptance, and reject
changes that improve an isolated kernel but not end-to-end decoding.

When changing target layers, caches, quantization, sampling, batching, or
chunked prefill, rerun both synthetic rollback tests and real-model exactness.
Treat a new checkpoint layout or architecture tag as a compatibility change,
not as proof that an existing adapter applies.

For rotating caches, test beyond the window boundary and across repeated
accept/reject rounds. Restoring only the cursor is insufficient after eviction
or rotation. Transactions retain the serving window and incoming KV, then replay
the accepted updates in their original order. Test batch rows independently:
an incorrect stride in an intermediate-state kernel can write beyond its output
allocation even when batch-one tests pass.

## Runtime block sizes

GLM-5-Next starts at its native MTP depth and can extend to a three-token block
(two proposals) when the shared acceptance policy observes reliable acceptance.
This reduced round overhead in the GLM-5.3-Flash greedy batch-one and batch-four
checks. Batched stochastic sampling keeps a two-token ceiling by default because
the extra proposal did not repay its verification cost in that workload.
Explicit `runtime_block_size` and `--draft-block-size` settings take precedence.

DeepSeek-V4 DSpark defaults to a two-token block (one proposal). Its ordinary
operators preserve decode arithmetic and physical attention-window order. Larger
blocks can cost more than their accepted tokens save. The current exact path
remains slower than baseline on the measured M3 Ultra workloads; use ordinary
decode when throughput is the priority. Text prefill remains chunked with the
drafter attached; captured features are retained across chunks. Image spans
still use the model's whole-image prefill policy.
