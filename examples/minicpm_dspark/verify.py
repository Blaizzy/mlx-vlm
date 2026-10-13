"""Full-checkpoint logit/hidden/cache qualification for the MiniCPM adapter."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
from adapter import install_padded_head

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.utils import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", type=Path, default=Path("minicpm-verifier-checks.json"))
p.add_argument("--capture-layers", type=int, nargs="+")
p.add_argument(
    "--long-context", action="store_true", help="Also check the 64K boundary"
)
a = p.parse_args()
model, _ = load(a.model)
lm = model.language_model
install_padded_head(lm)
results = []
cases = [
    (32, 1, 0),
    (32, 5, 3),
    (32, 8, 8),
    (32, 9, 4),
    (1020, 8, 3),
    (1020, 9, 0),
    (2048, 6, 2),
    (16376, 8, 3),
    (16378, 8, 3),
]
if a.long_context:
    cases.extend([(65528, 8, 3), (65530, 8, 3)])
for prefix_length, rows, accepted in cases:
    cache = lm.make_cache()
    prefix = (mx.arange(prefix_length)[None] % 100) + 100
    lm(prefix, cache=cache, skip_logits=True)
    mx.eval([layer.state for layer in cache])
    reference = deepcopy(cache)
    retained = deepcopy(cache)
    ids = (mx.arange(rows)[None] * 2 + 1) + 200
    options = dict(
        return_hidden=True,
        capture_layer_ids=a.capture_layers or list(range(3, len(lm.model.layers), 4)),
    )
    grouped = lm(ids, cache=cache, speculative_verify=True, **options)
    mx.eval(grouped.logits)
    steps = []
    for i in range(rows):
        step = lm(ids[:, i : i + 1], cache=reference, **options)
        mx.eval(step.logits)
        steps.append(step)
        if i + 1 == accepted:
            retained = deepcopy(reference)
    serial = mx.concatenate([s.logits for s in steps], 1)
    logits_equal = mx.array_equal(grouped.logits, serial).item()
    hidden_equal = all(
        mx.array_equal(h, mx.concatenate([s.hidden_states[j] for s in steps], 1)).item()
        for j, h in enumerate(grouped.hidden_states)
    )
    grouped.gdn_states.commit([accepted])
    recurrent_equal = all(
        mx.array_equal(x, y).item()
        for actual, expected in zip(cache, retained)
        if isinstance(actual, ArraysCache)
        for x, y in zip(actual.state, expected.state)
    )
    kv_equal = True
    for actual, expected in zip(cache, retained):
        if not isinstance(actual, ArraysCache):
            kv_equal &= actual.offset == expected.offset
            for x, y in zip(actual.state, expected.state):
                kv_equal &= mx.array_equal(
                    x[..., : actual.offset, :], y[..., : expected.offset, :]
                ).item()
    probe = mx.array([[333]], mx.int32)
    next_equal = mx.array_equal(
        lm(probe, cache=cache).logits, lm(probe, cache=retained).logits
    ).item()
    result = dict(
        prefix=prefix_length,
        rows=rows,
        accepted=accepted,
        logits_equal=logits_equal,
        hidden_equal=hidden_equal,
        recurrent_equal=recurrent_equal,
        kv_equal=kv_equal,
        next_equal=next_equal,
        max_logit_error=mx.max(
            mx.abs(grouped.logits.astype(mx.float32) - serial.astype(mx.float32))
        ).item(),
    )
    results.append(result)
    print(json.dumps(result), flush=True)
    a.output.write_text(json.dumps(results, indent=2))
assert all(
    all(
        r[k]
        for k in (
            "logits_equal",
            "hidden_equal",
            "recurrent_equal",
            "kv_equal",
            "next_equal",
        )
    )
    for r in results
)
