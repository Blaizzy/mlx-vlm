---
name: benchmarking
description: Use this skill when the user wants to benchmark an MLX-VLM change and present the numbers in a PR — fork-vs-main A/B comparisons, isolated-module micro-benchmarks, median-of-N timing with warmup, peak-memory reporting, correctness checks, parameter sweeps, and self-contained reproducible bench scripts to paste into a PR description.
---

# Benchmarking & PR A/B Testing

Use this workflow to produce **credible, reproducible** performance numbers for a PR.

## Rules for a trustworthy benchmark

- **Median of N** (not mean), with a few warmup iterations discarded.
- Call `mx.eval(...)` / `mx.synchronize()` before stopping the timer (MLX is lazy — untimed work hides otherwise).
- Report **peak memory** via `mx.get_peak_memory()/1e9` (GB); reset with `mx.reset_peak_memory()` if available.
- Include an **inline correctness check** — a benchmark that computes the wrong thing is meaningless. Assert the new path matches the baseline (exactly in the degenerate limit, or within tolerance).
- **Sweep one parameter** (sequence length, batch, tokens, layers) and print a **markdown table** with `flush=True`.
- Prefer a **random-init synthetic model** (tiny config) so the script needs no checkpoint download and runs on a fresh checkout.
- State the hardware (Apple-Silicon chip + RAM). A model must fit in RAM; isolate the module under test with random weights if the full model won't fit — don't run large models on an 8 GB machine.

## Fork-vs-main A/B (paste into the PR body)

Clones upstream `main` and your fork, runs the *identical* bench on both, and joins into a speedup table:

```bash
#!/usr/bin/env bash
# <feature> perf: upstream main vs fork, random-init tiny <model>, swept over <param>.
set -euo pipefail
W=$(mktemp -d)
git clone -q --depth 1 https://github.com/Blaizzy/mlx-vlm "$W/main"
git clone -q --depth 1 https://github.com/<you>/mlx-vlm   "$W/fork"

cat > "$W/b.py" <<'PY'
import time, numpy as np, mlx.core as mx
from mlx_vlm.models.<model> import Model, ModelConfig
cfg = ModelConfig(...tiny config...)          # small enough to fit in RAM
m = Model(cfg); m.eval(); mx.eval(m.parameters())
def ms(T):
    c = m.make_cache(); p = 0
    while p < T:                               # chunked prefill avoids OOM
        n = min(256, T - p)
        mx.eval(m(mx.array(np.random.randint(0, cfg.vocab_size, (1, n))), cache=c).logits); p += n
    for _ in range(4): mx.eval(m(mx.array([[0]]), cache=c).logits)   # warmup
    t = time.perf_counter()
    for _ in range(20): mx.eval(m(mx.array([[0]]), cache=c).logits)
    return (time.perf_counter() - t) / 20 * 1e3
for T in (16384, 32768, 65536):
    print(T, round(ms(T), 2), flush=True)
PY

run() { uv venv -q "$1/.venv"; uv pip install -q -e "$1" --python "$1/.venv/bin/python"; "$1/.venv/bin/python" "$W/b.py"; }
run "$W/main" > "$W/m.txt"
run "$W/fork" > "$W/f.txt"
echo; printf "| param | main | fork | speedup |\n|--:|--:|--:|--:|\n"
join "$W/m.txt" "$W/f.txt" | awk '{printf "| %d | %.2f ms | %.2f ms | %.2fx |\n", $1, $2, $3, $2/$3}'
```

## Isolated-module micro-benchmark

For one function/kernel (time + peak mem + correctness), swept over a param:

```python
import time, statistics, mlx.core as mx
def bench(fn, it=30, warm=5):
    for _ in range(warm): mx.eval(fn())
    mx.synchronize()
    if hasattr(mx, "reset_peak_memory"): mx.reset_peak_memory()
    s = []
    for _ in range(it):
        t0 = time.perf_counter(); mx.eval(fn()); s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s), mx.get_peak_memory() / 1e9
# build inputs; assert new == baseline in the degenerate limit; then print a swept markdown table.
```

## Presenting in the PR

- Paste the **exact script** (self-contained) plus the resulting **markdown table**, and state the chip/RAM and mlx-vlm commit.
- Report both **latency** and **peak memory** — a speedup that blows the memory budget isn't a win.
- If comparing against another PR (e.g. an earlier version), link it and use the same script for both.
- Keep the repro script in scratch (not committed); only the script text + table go in the PR body.
