"""Expert-offload: page routed-MoE experts from an SSD store instead of holding
them resident, so a checkpoint bigger than RAM can still run.

``repack`` splits each MoE layer's stacked expert tensor into a per-expert
on-disk store; ``ExpertStore`` mmaps it and keeps a byte-budgeted LRU of
resident experts, evicting the coldest ones once touched experts approach the
GPU's recommended working set; ``patch_model`` swaps every switch layer
(``SwitchGLU``/subclasses, or a fused ``gate_up_proj`` variant like
Laguna/MiniMax-M3-VL) for an ``OffloadedSwitchGLU`` that computes only the
router-selected experts. Loading is transparent: ``mlx_vlm.load()`` detects
``offload_index.json`` and patches automatically. Prefill must be chunked
(``prefill_step_size``), or a lazy full-prompt forward pins every expert
across every layer in one graph until the final eval and OOMs anyway.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from collections import OrderedDict
from typing import Optional, Tuple

# stacked: switch_mlp.gate_proj.weight [E,out,in]; per-expert: experts.{j}.gate_proj.weight;
# stacked-fused: switch_mlp.gate_up_proj.weight [E,2*out,in], gate = first half of axis 1.
_PROJ = r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|scales|biases)$"
_FUSED_PROJ = r"gate_up_proj\.(?P<kind>weight|scales|biases)$"
PEREXPERT_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\..*?experts\.(?P<j>\d+)\." + _PROJ
)
STACKED_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\..*?(?:experts|switch_mlp)\." + _PROJ
)
STACKED_FUSED_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\..*?(?:experts|switch_mlp)\." + _FUSED_PROJ
)


def plan(tensor_names) -> dict:
    """Pure partition of names -> resident vs per-layer routed experts.

    experts[layer] is a list of (store_key_or_None, source_name, mode):
    per-expert -> (key, name, None) copied as-is; stacked/fused -> (None, name,
    "STACK"/"STACK_FUSED") sliced (and, for fused, split gate/up) at repack.
    """
    resident, experts = [], {}
    for name in tensor_names:
        if "shared_expert" in name:  # shared_expert / shared_experts -> resident
            resident.append(name)
            continue
        m = PEREXPERT_RE.match(name)
        if m:
            key = f"e{int(m['j'])}.{m['proj']}.{m['kind']}"
            experts.setdefault(int(m["layer"]), []).append((key, name, None))
            continue
        m = STACKED_FUSED_RE.match(name)
        if m:
            experts.setdefault(int(m["layer"]), []).append((None, name, "STACK_FUSED"))
            continue
        m = STACKED_RE.match(name)
        if m:
            experts.setdefault(int(m["layer"]), []).append((None, name, "STACK"))
            continue
        resident.append(name)
    return {"resident": sorted(resident), "experts": experts, "layers": sorted(experts)}


def _check_disk_headroom(build: str, out: str, margin: float = 2.0) -> None:
    """The source checkpoint and the growing offload dir coexist on disk for
    the whole repack, so free space needs to cover roughly ``margin`` times
    the source size. Failing this mid-write can silently truncate output
    rather than raise, so check up front instead."""
    source_bytes = sum(
        os.path.getsize(f) for f in glob.glob(os.path.join(build, "*.safetensors"))
    )
    free_bytes = shutil.disk_usage(out).free
    needed_bytes = source_bytes * margin
    if free_bytes < needed_bytes:
        raise ValueError(
            f"Not enough disk space to repack {build!r}: source checkpoint is "
            f"{source_bytes / 1e9:.1f} GB, repack needs roughly {margin:.0f}x that "
            f"({needed_bytes / 1e9:.1f} GB) free at {out!r}, but only "
            f"{free_bytes / 1e9:.1f} GB is available."
        )


def repack(build: str, out: str, resident_shard_gb: float = 5.0) -> None:
    """Memory-bounded repack (streams shard-by-shard so it runs on constrained
    RAM, e.g. a 16 GB mini). Uses the safetensors index if present, else globs
    ``*.safetensors``, mirroring ``mlx_vlm.utils.load_model``'s own fallback.
    """
    import gc

    import mlx.core as mx

    os.makedirs(os.path.join(out, "experts"), exist_ok=True)
    _check_disk_headroom(build, out)

    idx_path = os.path.join(build, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
    else:
        files = [
            f
            for f in glob.glob(os.path.join(build, "*.safetensors"))
            if not f.endswith("consolidated.safetensors")
        ]
        wmap = {}
        for f in files:
            fname = os.path.basename(f)
            wmap.update({k: fname for k in mx.load(f)})
    p = plan(list(wmap))
    resident_set = set(p["resident"])

    def clear():
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    # ---- RESIDENT: stream shards -> sharded resident-*.safetensors (peak ~ one shard + buffer)
    by_shard = {}
    for n in wmap:
        by_shard.setdefault(wmap[n], []).append(n)
    buf, buf_bytes, ri, res_index = {}, 0, 0, {}

    def flush_resident():
        nonlocal buf, buf_bytes, ri
        if not buf:
            return
        ri += 1
        fn = f"resident-{ri:04d}.safetensors"
        mx.eval(list(buf.values()))
        mx.save_safetensors(os.path.join(out, fn), buf, metadata={"format": "mlx"})
        for k in buf:
            res_index[k] = fn
        buf, buf_bytes = {}, 0
        clear()

    for shard, tns in by_shard.items():
        need = [n for n in tns if n in resident_set]
        if not need:
            continue
        w = mx.load(os.path.join(build, shard))
        for n in need:
            buf[n] = w[n]
            buf_bytes += w[n].nbytes
            if buf_bytes >= resident_shard_gb * 1e9:
                flush_resident()
        del w
        clear()
    flush_resident()
    print(f"resident: {len(res_index)} tensors in {ri} shards", flush=True)

    # ---- EXPERTS: per layer, load only the needed shards, split, free (peak ~ a few shards)
    n_experts = None
    for lid, entries in sorted(p["experts"].items()):
        layer = {}
        for key, src, mode in entries:
            w = mx.load(os.path.join(build, wmap[src]))  # re-mmap on demand
            arr = w[src]
            if mode == "STACK":
                E = arr.shape[0]
                n_experts = max(n_experts or 0, E)
                mm = STACKED_RE.match(src)
                for j in range(E):
                    layer[f"e{j}.{mm['proj']}.{mm['kind']}"] = arr[j]
            elif mode == "STACK_FUSED":
                # gate = first half of axis 1 (the doubled output dim), up = second.
                E = arr.shape[0]
                half = arr.shape[1] // 2
                n_experts = max(n_experts or 0, E)
                mm = STACKED_FUSED_RE.match(src)
                kind = mm["kind"]
                gate_half, up_half = arr[:, :half, ...], arr[:, half:, ...]
                for j in range(E):
                    layer[f"e{j}.gate_proj.{kind}"] = gate_half[j]
                    layer[f"e{j}.up_proj.{kind}"] = up_half[j]
            else:
                layer[key] = arr
                n_experts = max(n_experts or 0, int(key[1:].split(".")[0]) + 1)
            del w
        mx.eval(list(layer.values()))
        mx.save_safetensors(
            os.path.join(out, "experts", f"layer_{lid:04d}.safetensors"),
            layer,
            metadata={"format": "mlx"},
        )
        layer = None
        clear()
        if lid % 8 == 0:
            print(f"  experts: layer {lid} done", flush=True)
    print(f"experts: {len(p['layers'])} MoE layers x {n_experts} experts", flush=True)

    import shutil

    for fn in os.listdir(build):  # passthrough config/tokenizer/processor/code
        src = os.path.join(build, fn)
        if (
            os.path.isfile(src)
            and not fn.startswith("model-")
            and not fn.endswith(".safetensors")
        ):
            shutil.copy2(src, os.path.join(out, fn))
        elif os.path.isdir(src) and not fn.startswith(".") and fn != "experts":
            shutil.copytree(
                src,
                os.path.join(out, fn),
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    json.dump(
        {"layers": p["layers"], "num_experts": n_experts},
        open(os.path.join(out, "offload_index.json"), "w"),
        indent=2,
    )


_PROJ_KEYS = tuple(
    f"{p}.{k}"
    for p in ("gate_proj", "up_proj", "down_proj")
    for k in ("weight", "scales", "biases")
)


def _resident_bytes_on_disk(offload_dir: str) -> int:
    return sum(
        os.path.getsize(f)
        for f in glob.glob(os.path.join(offload_dir, "resident-*.safetensors"))
    )


def _kv_cache_bytes(cache_entries) -> int:
    from mlx.utils import tree_flatten

    total = 0
    for c in cache_entries:
        children = getattr(c, "caches", None)
        if children is not None:
            total += _kv_cache_bytes(children)
            continue
        state = getattr(c, "state", None)
        if state is None:
            continue
        for _, arr in tree_flatten(state):
            if hasattr(arr, "nbytes"):
                total += arr.nbytes
    return total


def _kv_cache_token_caps(cache_entries) -> list:
    caps = []
    for c in cache_entries:
        children = getattr(c, "caches", None)
        if children is not None:
            caps.extend(_kv_cache_token_caps(children))
            continue
        caps.append(getattr(c, "max_size", None))
    return caps


def _estimate_kv_reserve_bytes(model, max_kv_size: Optional[int]) -> int:
    """Reserve headroom for KV cache growth up to ``max_kv_size`` tokens, so
    the expert budget doesn't overcommit against a model whose cache grows
    with context (a plain ``KVCache`` has no compression or window, unlike
    DeepSeek's ``PoolingCache``/``RotatingKVCache``). Measured empirically by
    running two real tiny forward passes through the model's own
    ``make_cache()`` and reading the marginal per-token growth off
    ``cache.state`` -- generic across cache classes (works whether the cache
    is bounded, compressed, or plain) rather than guessing a per-architecture
    formula. Returns 0 (no reservation -- today's behavior) if anything about
    the model's calling convention doesn't cooperate, or ``max_kv_size`` is
    unset; a wrong reservation is worse than none.
    """
    if not max_kv_size or max_kv_size <= 0:
        return 0
    import mlx.core as mx

    lm = getattr(model, "language_model", model)
    make_cache = getattr(lm, "make_cache", None)
    if not callable(make_cache):
        return 0
    try:
        cache = make_cache()
        ids = mx.zeros((1, 3), dtype=mx.int32)
        lm(ids[:, :2], cache=cache)
        mx.eval([c.state for c in cache])
        bytes_at_2 = _kv_cache_bytes(cache)
        lm(ids[:, 2:3], cache=cache)
        mx.eval([c.state for c in cache])
        bytes_at_3 = _kv_cache_bytes(cache)
    except Exception:
        return 0

    per_token = max(0, bytes_at_3 - bytes_at_2)
    caps = _kv_cache_token_caps(cache)
    effective_tokens = max_kv_size
    if caps and all(c is not None for c in caps):
        effective_tokens = min(max_kv_size, max(int(c) for c in caps))
    return bytes_at_3 + per_token * max(0, effective_tokens - 3)


def _default_expert_cache_bytes(
    resident_bytes: int = 0, kv_reserve_bytes: int = 0
) -> int:
    import mlx.core as mx

    try:
        recommended = mx.device_info()["max_recommended_working_set_size"]
    except Exception:
        return 0
    budget = int(0.8 * recommended) - resident_bytes - kv_reserve_bytes
    return max(0, budget)


class ExpertStore:
    """Serves per-expert quantized weights from memory-mapped per-layer files.

    A materialized expert stays resident in ``self._maps[layer_id]`` once
    touched -- an earlier version relied on that alone (measured: an
    additional LRU cache had a 0% hit rate for single-request serving, since
    ``mx.load()``'s mmap plus the OS page cache already serves repeat reads).
    That held for every model this was tested against, where the whole
    routed-expert corpus fits in RAM regardless. It does not hold at DeepSeek
    V4 Flash's scale (256 experts x 43 layers): routing alone samples close
    to the full expert set within a couple hundred tokens (prompt and
    generated combined), so unbounded residency grows toward the full
    on-disk corpus and OOMs once that exceeds physical memory. ``get()`` now
    tracks per-``(layer, expert)`` residency with a byte budget and evicts
    the least-recently-used expert -- by dropping its keys from ``self._maps``
    so MLX can reclaim the buffers, not by discarding the per-layer mmap
    handle itself -- re-fetching lazily from the same on-disk file on the
    next miss. Concurrent ``get()`` calls still need no locking for the
    reads themselves, but eviction mutates ``self._maps`` and ``self._lru``,
    so callers serving multiple requests concurrently must serialize access
    to a single ``ExpertStore``. That does not make the returned arrays safe
    to *evaluate* from a different thread than the one that called
    ``mx.load()`` here -- MLX raises ``RuntimeError: no Stream(...) in
    current thread`` for that, a pre-existing constraint of ``mx.load()``
    generally, not new here or fixable by locking.
    """

    def __init__(
        self,
        offload_dir: str,
        expert_cache_bytes: Optional[int] = None,
        kv_reserve_bytes: int = 0,
    ):
        import mlx.core as mx

        idx = json.load(open(os.path.join(offload_dir, "offload_index.json")))
        self.num_experts = idx["num_experts"]
        self._paths = {}
        self._maps = {}  # {layer_id: {name: lazy mx.array}}
        for path in glob.glob(
            os.path.join(offload_dir, "experts", "layer_*.safetensors")
        ):
            lid = int(os.path.basename(path).split("_")[1].split(".")[0])
            self._paths[lid] = path
            self._maps[lid] = mx.load(path)  # mmap, lazy
        self._budget = (
            expert_cache_bytes
            if expert_cache_bytes is not None
            else _default_expert_cache_bytes(
                resident_bytes=_resident_bytes_on_disk(offload_dir),
                kv_reserve_bytes=kv_reserve_bytes,
            )
        )
        self._lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def experts_present(self, layer_id: int) -> bool:
        return layer_id in self._maps

    def _ensure_loaded(self, layer_id: int, j: int, m: dict) -> None:
        import mlx.core as mx

        if f"e{j}.gate_proj.weight" in m:
            return
        fresh = mx.load(self._paths[layer_id])
        for k in _PROJ_KEYS:
            name = f"e{j}.{k}"
            if name in fresh:
                m[name] = fresh[name]

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        import mlx.core as mx

        if self._budget <= 0:
            return
        evicted = False
        while self._lru and self._resident_bytes + incoming_bytes > self._budget:
            (lid, j), nbytes = self._lru.popitem(last=False)
            m = self._maps[lid]
            for k in _PROJ_KEYS:
                m.pop(f"e{j}.{k}", None)
            self._resident_bytes -= nbytes
            self._evictions += 1
            evicted = True
        if evicted:
            try:
                mx.clear_cache()
            except Exception:
                pass

    def get(self, layer_id: int, j: int):
        """(gate, up, down), each (w, scales, biases) lazy arrays for expert j.
        ``scales``/``biases`` are ``None`` for an unquantized (plain
        bf16/float32) expert -- ``e{j}.{proj}.weight`` is the only key repack
        wrote for it."""
        m = self._maps[layer_id]
        self._ensure_loaded(layer_id, j, m)
        key = (layer_id, j)
        if key in self._lru:
            self._lru.move_to_end(key)
            self._hits += 1
        else:
            self._misses += 1
            nbytes = sum(m[f"e{j}.{k}"].nbytes for k in _PROJ_KEYS if f"e{j}.{k}" in m)
            self._evict_until_fits(nbytes)
            self._lru[key] = nbytes
            self._resident_bytes += nbytes
        trip = lambda p: (
            m[f"e{j}.{p}.weight"],
            m.get(f"e{j}.{p}.scales"),
            m.get(f"e{j}.{p}.biases"),
        )
        return (trip("gate_proj"), trip("up_proj"), trip("down_proj"))

    def stats(self) -> dict:
        """A snapshot of eviction behavior, for a server-side observability
        endpoint or ad hoc tuning of ``expert_cache_gb``. ``hit_rate`` is
        fraction of ``get()`` calls that found the expert already resident;
        a low rate under real traffic (not this session's cold start) means
        the budget is too small for the working set and thrashing."""
        total = self._hits + self._misses
        return {
            "budget_bytes": self._budget,
            "resident_bytes": self._resident_bytes,
            "resident_experts": len(self._lru),
            "num_experts": self.num_experts,
            "num_layers": len(self._maps),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": (self._hits / total) if total else None,
        }


def patch_model(
    model,
    offload_dir: str,
    expert_cache_gb: Optional[float] = None,
    max_kv_size: Optional[int] = None,
) -> "ExpertStore":
    """Swap every switch layer in ``model`` for an offloaded one (see module
    docstring for separate-vs-fused handling). group_size/bits/mode are
    resolved per projection via ``_quantization_for_path``, the same
    per-path override mechanism ``load_model``/``convert`` use. Raises if no
    expert files are found or nothing gets swapped, rather than silently
    loading fully resident. ``expert_cache_gb`` bounds the store's resident
    expert set (default: 80% of the GPU's recommended working set, minus the
    on-disk resident weight size and minus a KV-cache reserve for
    ``max_kv_size`` tokens if given) -- pass a smaller value to leave more
    headroom for a large KV cache, or a very large one to effectively
    disable eviction for a checkpoint already known to fit. ``max_kv_size``
    only affects the *auto* budget; it's ignored if ``expert_cache_gb`` is
    set explicitly. Returns the store.
    """
    import mlx.nn as nn

    from .models.switch_layers import OffloadedSwitchGLU
    from .quantization.one_bit import _quantization_for_path

    cfg = json.load(open(os.path.join(offload_dir, "config.json")))
    quantization = (
        cfg.get("quantization") or cfg.get("text_config", {}).get("quantization") or {}
    )
    default_quant = {
        "group_size": quantization.get("group_size", 64),
        "bits": quantization.get("bits", 4),
        "mode": quantization.get("mode", "affine"),
    }

    def resolve_quant(path: str) -> Tuple[int, int, str]:
        merged = {**default_quant, **_quantization_for_path(quantization, path)}
        return merged["group_size"], merged["bits"], merged["mode"]

    expert_cache_bytes = (
        int(expert_cache_gb * 1e9) if expert_cache_gb is not None else None
    )
    kv_reserve_bytes = (
        0
        if expert_cache_bytes is not None
        else _estimate_kv_reserve_bytes(model, max_kv_size)
    )
    store = ExpertStore(offload_dir, expert_cache_bytes, kv_reserve_bytes)
    if not store._maps:
        raise ValueError(
            f"No expert files found under {offload_dir}/experts -- this "
            "doesn't look like a valid repack() output directory."
        )
    swapped = [0]
    missing_layers = []

    def visit(module, path=""):
        for name, child in list(module.items()):
            cp = f"{path}.{name}" if path else name
            if isinstance(child, nn.Module):
                # Duck-typed (not isinstance SwitchGLU) to cover fused
                # variants (Laguna/MiniMax-M3-VL) without model-specific imports.
                is_separate = all(
                    hasattr(child, p) for p in ("gate_proj", "up_proj", "down_proj")
                )
                is_fused = hasattr(child, "gate_up_proj") and hasattr(
                    child, "down_proj"
                )
                if (is_separate or is_fused) and _n_experts(child) == store.num_experts:
                    # A different expert count (e.g. always-on shared experts as
                    # their own switch layer) stays resident -- else index collision.
                    lid = _layer_id(cp)
                    if lid is not None and not store.experts_present(lid):
                        # Shape/expert-count matches, so this layer WAS meant to
                        # be offloaded -- a missing experts/layer_*.safetensors
                        # here means a partial or corrupted repack(), not an
                        # intentional skip. Left un-swapped, it would silently
                        # run on random-init weights with no error anywhere.
                        missing_layers.append((lid, cp))
                    elif lid is not None:
                        if is_separate:
                            gate_quant = resolve_quant(f"{cp}.gate_proj")
                            up_quant = resolve_quant(f"{cp}.up_proj")
                            # SwitchLinear's optional per-expert additive bias, loaded
                            # resident (tiny) before this module is replaced.
                            gate_bias = getattr(child.gate_proj, "bias", None)
                            up_bias = getattr(child.up_proj, "bias", None)
                        else:
                            # Fused: gate/up are one on-disk tensor, quantized as a
                            # single unit, so they share one resolved quant triple.
                            gate_quant = up_quant = resolve_quant(f"{cp}.gate_up_proj")
                            fused_bias = getattr(child.gate_up_proj, "bias", None)
                            if fused_bias is not None:
                                half = fused_bias.shape[-1] // 2
                                gate_bias = fused_bias[..., :half]
                                up_bias = fused_bias[..., half:]
                            else:
                                gate_bias = up_bias = None
                        module[name] = OffloadedSwitchGLU(
                            store,
                            lid,
                            gate_quant,
                            up_quant,
                            resolve_quant(f"{cp}.down_proj"),
                            activation=getattr(child, "activation", None),
                            gate_scale=getattr(child, "gate_scale", None),
                            out_scale=getattr(child, "out_scale", None),
                            gate_bias=gate_bias,
                            up_bias=up_bias,
                            down_bias=getattr(child.down_proj, "bias", None),
                        )
                        swapped[0] += 1
                else:
                    visit(child, cp)
            elif isinstance(child, list):
                for i, c in enumerate(child):
                    if isinstance(c, nn.Module):
                        visit(c, f"{cp}.{i}")
            elif isinstance(child, dict):
                for k, c in child.items():
                    if isinstance(c, nn.Module):
                        visit(c, f"{cp}.{k}")

    visit(model)
    if missing_layers:
        detail = ", ".join(f"layer {lid} ({cp})" for lid, cp in sorted(missing_layers))
        raise ValueError(
            f"Offload dir {offload_dir} is missing experts/layer_*.safetensors for "
            f"{len(missing_layers)} switch layer(s) whose shape and expert count "
            f"match this model's routed experts: {detail}. This looks like a "
            "partial or corrupted repack() output -- re-run repack() rather than "
            "silently running those layers on random-init weights."
        )
    if swapped[0] == 0:
        raise ValueError(
            f"patch_model swapped 0 modules for {offload_dir} -- offload_index.json "
            "declares experts but no matching switch layer (with a matching expert "
            "count) was found anywhere in the model."
        )
    store.swapped = swapped[0]
    return store


def _layer_id(path: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", path)
    return int(m.group(1)) if m else None


def _n_experts(switch_glu) -> int:
    """Number of experts in a switch layer -- its ``gate_proj`` (or, for a
    fused switch layer, ``gate_up_proj``) stacks per-expert weights as
    ``[E, out, in]`` (``[E, out, in//pack]`` when quantized), so axis 0 is
    the expert count. Falls back to 0 for an already-offloaded or otherwise
    nonstandard module (no ``.weight`` on either)."""
    w = getattr(getattr(switch_glu, "gate_proj", None), "weight", None)
    if w is None:
        w = getattr(getattr(switch_glu, "gate_up_proj", None), "weight", None)
    return int(w.shape[0]) if w is not None else 0


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Repack an mlx-vlm MoE checkpoint for expert-offload."
    )
    ap.add_argument("--build", required=True, help="mlx-vlm checkpoint directory")
    ap.add_argument("--out", required=True, help="output offload directory")
    a = ap.parse_args()
    repack(a.build, a.out)


if __name__ == "__main__":
    main()
