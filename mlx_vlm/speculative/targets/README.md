# Speculative targets

Bind a loaded language model with `bind_speculative_target` once per generation
session, after loading, quantizing, and casting its weights. GLM-5-Next uses an
isolated execution view: it copies module containers, shares parameter arrays,
and specializes projections, temporal state capture, sparse attention execution,
and fused hyperconnection operations. The model's normal decoder traversal owns
the architecture. Changes to that traversal apply to both execution paths.

The serving model has no speculative methods or flags. Its normal calls also
remain available through the bound target. Other architectures retain their
existing target hooks while migrating to this boundary.

DeepSeek-V4's DSpark drafter uses the DFlash runtime. Its bound target starts
pooling-cache transactions before verification, preserves native projection and
attention reduction order, and returns the transaction with the captured
features. Its ordinary attention classes share one projection/output traversal;
the bound view specializes the attention operation. `SpeculativePrefill`
retains features across prompt chunks so attaching the drafter does not require
a full-context forward.

The runtime normalizes legacy output tuples into `_MTPVerifyResult.rollback_state`.
Keep forward evaluation, sampling, acceptance, and draft updates inside the
round's error boundary. Commit every successful round before yielding tokens,
including rounds where every draft token was accepted. Abort unfinished target
and draft transactions on failure.

`SpeculativeCacheTransaction.commit(retained_lengths)` owns both temporal state
and append-cache trimming. Rotating caches retain a bounded window snapshot;
partial commits restore that window and append only accepted keys and values.
Incoming updates are recorded before eviction, so this also handles multiple
one-token appends and wraparound within one verification round.
Lengths count verified input positions retained per
row, including the previous bonus token. The runtime converts accepted draft
counts to these lengths exactly once. Drafters use their actual appended lengths.

Projection contracts are explicit in `speculative/ops/linear.py`:
`native_batch_linear` matches independent B×1 forwards, while the existing Qwen
operations preserve singleton-row arithmetic. Quantized kernels keep their
shape and dtype guards; unsupported shapes use the corresponding reference
fallback. These contracts must not be exchanged merely because both produce
similar logits.
