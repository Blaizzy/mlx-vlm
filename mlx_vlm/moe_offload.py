"""Expert-offload: page routed-MoE experts from an SSD store instead of holding
them resident, so a checkpoint bigger than RAM can still run.

An MoE only fires a few experts per token (e.g. 6 of 256), yet the experts are
the overwhelming majority of the weights. ``repack`` splits each MoE layer's
stacked expert tensor into a per-expert on-disk store; ``ExpertStore`` mmaps
it; ``patch_model`` swaps every ``SwitchGLU`` in a loaded model for an
``OffloadedSwitchGLU`` that computes only the router-selected experts, paged
from that store. Works with any model built on
``mlx_vlm.models.switch_layers.SwitchGLU`` (or a subclass of it, e.g. the
native Inkling model) -- most MoE models in this repo, but not all: a few
(e.g. Laguna, MiniMax-M3-VL) use a *fused* switch layer with a single
``gate_up_proj`` instead of separate ``gate_proj``/``up_proj`` modules, and
aren't supported yet. ``patch_model`` raises rather than silently loading
those fully resident.

Loading an offload dir (a directory produced by ``repack``) is transparent:
``mlx_vlm.load()`` detects ``offload_index.json`` and patches automatically.

Prefill must be chunked (``prefill_step_size`` in ``mlx_vlm.generate``), or a
lazy full-prompt forward pins every routed expert across every layer in one
graph until the final eval -- on a large MoE that OOMs regardless of offload.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
from collections import OrderedDict
from typing import Optional, Tuple

# Routed-expert weights of an MoE MLP block. Two on-disk conventions are
# supported:
#   stacked    : ...switch_mlp.gate_proj.weight        (one [E,out,in] tensor)
#   per-expert : ...experts.{j}.gate_proj.weight        (one tensor per expert)
# shared experts (...shared_expert(s)...) always stay resident.
_PROJ = r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|scales|biases)$"
PEREXPERT_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\..*?experts\.(?P<j>\d+)\." + _PROJ
)
STACKED_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\..*?(?:experts|switch_mlp)\." + _PROJ
)


def plan(tensor_names) -> dict:
    """Pure partition of names -> resident vs per-layer routed experts (unit-testable).

    experts[layer] is a list of (store_key_or_None, source_name, expert_idx):
      - per-expert source: (``e{j}.{proj}.{kind}``, name, None)  -> copied as-is
      - stacked source:    (None,                    name, "STACK") -> sliced per expert at repack
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
        m = STACKED_RE.match(name)
        if m:
            experts.setdefault(int(m["layer"]), []).append((None, name, "STACK"))
            continue
        resident.append(name)
    return {"resident": sorted(resident), "experts": experts, "layers": sorted(experts)}


def repack(build: str, out: str, resident_shard_gb: float = 5.0) -> None:
    """Memory-bounded repack (streams shard-by-shard so it runs on constrained
    RAM, e.g. a 16 GB mini). Uses the safetensors index if present, else globs
    ``*.safetensors``, mirroring ``mlx_vlm.utils.load_model``'s own fallback.
    """
    import gc

    import mlx.core as mx

    os.makedirs(os.path.join(out, "experts"), exist_ok=True)

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


class ExpertStore:
    """Serves per-expert quantized weights from memory-mapped per-layer files."""

    def __init__(self, offload_dir: str, lru_experts: Optional[int] = None):
        import mlx.core as mx

        idx = json.load(open(os.path.join(offload_dir, "offload_index.json")))
        self.num_experts = idx["num_experts"]
        self._maps = {}  # {layer_id: {name: lazy mx.array}}
        for path in glob.glob(
            os.path.join(offload_dir, "experts", "layer_*.safetensors")
        ):
            lid = int(os.path.basename(path).split("_")[1].split(".")[0])
            self._maps[lid] = mx.load(path)  # mmap, lazy
        self._lru: OrderedDict = OrderedDict()
        self._cap = lru_experts
        self.hits = self.misses = 0
        # Guards _lru/hits/misses: batched/server-style generation can call
        # get() from multiple threads against one shared model instance.
        self._lock = threading.Lock()

    def experts_present(self, layer_id: int) -> bool:
        return layer_id in self._maps

    def get(self, layer_id: int, j: int):
        """(gate, up, down), each (w, scales, biases) lazy arrays for expert j.
        ``scales``/``biases`` are ``None`` for an unquantized (plain
        bf16/float32) expert -- ``e{j}.{proj}.weight`` is the only key repack
        wrote for it."""
        key = (layer_id, j)
        with self._lock:
            if key in self._lru:
                self.hits += 1
                self._lru.move_to_end(key)
                return self._lru[key]
            self.misses += 1
        m = self._maps[layer_id]
        trip = lambda p: (
            m[f"e{j}.{p}.weight"],
            m.get(f"e{j}.{p}.scales"),
            m.get(f"e{j}.{p}.biases"),
        )
        val = (trip("gate_proj"), trip("up_proj"), trip("down_proj"))
        if self._cap:
            with self._lock:
                self._lru[key] = val
                if len(self._lru) > self._cap:
                    self._lru.popitem(last=False)
        return val

    def stats(self) -> dict:
        tot = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / tot, 4) if tot else 0.0,
        }


def patch_model(
    model, offload_dir: str, lru_experts: Optional[int] = None
) -> "ExpertStore":
    """Swap every ``SwitchGLU`` (or subclass, e.g. Inkling's) in ``model`` for an
    offloaded one, paged from ``offload_dir``. group_size/bits/mode are
    resolved *per projection* from ``offload_dir/config.json``'s
    ``quantization`` dict via the same per-path override mechanism
    ``mlx_vlm.utils.load_model`` and ``mlx_vlm.convert``'s mixed-precision
    recipes already use (a single MoE layer can legitimately have different
    bits per projection today -- e.g. a ``mixed_4_6`` conversion gives
    ``down_proj`` different bits than ``gate_proj``/``up_proj``). Returns the
    store so callers can inspect ``store.stats()``.

    Raises ``ValueError`` if no expert files are found in ``offload_dir``, or
    if nothing gets swapped -- a malformed/mistyped directory should fail
    loudly, not silently load fully resident with no offloading at all.
    Models built on a *fused* switch layer (a single ``gate_up_proj``, not
    separate ``gate_proj``/``up_proj`` modules -- e.g. Laguna, MiniMax-M3-VL)
    are not yet supported and will hit this same error, since their modules
    aren't ``SwitchGLU`` and their weights aren't split out by ``repack()``.

    ``mlx.nn.Module`` is a dict subclass -- child modules are dict items, so we
    traverse via ``module.items()`` and replace via ``module[name] = ...``
    (``setattr``/``vars`` do NOT reach them).
    """
    import mlx.nn as nn

    from .models.switch_layers import OffloadedSwitchGLU, SwitchGLU
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

    store = ExpertStore(offload_dir, lru_experts=lru_experts)
    if not store._maps:
        raise ValueError(
            f"No expert files found under {offload_dir}/experts -- this "
            "doesn't look like a valid repack() output directory."
        )
    swapped = [0]

    def visit(module, path=""):
        for name, child in list(module.items()):
            cp = f"{path}.{name}" if path else name
            if isinstance(child, nn.Module):
                if (
                    isinstance(child, SwitchGLU)
                    and _n_experts(child) == store.num_experts
                ):
                    # Only the ROUTED experts (count == store.num_experts) are
                    # offloaded. A sibling SwitchGLU-family module with a
                    # different expert count (e.g. always-on shared experts
                    # modeled as their own switch layer) stays resident --
                    # offloading it would collide its indices with the store.
                    lid = _layer_id(cp)
                    if lid is not None and store.experts_present(lid):
                        module[name] = OffloadedSwitchGLU(
                            store,
                            lid,
                            resolve_quant(f"{cp}.gate_proj"),
                            resolve_quant(f"{cp}.up_proj"),
                            resolve_quant(f"{cp}.down_proj"),
                            activation=getattr(child, "activation", None),
                            gate_scale=getattr(child, "gate_scale", None),
                            out_scale=getattr(child, "out_scale", None),
                            # SwitchLinear's optional per-expert additive bias
                            # (distinct from quantization scales/biases) --
                            # e.g. gpt-oss's experts carry one on every
                            # projection. Loaded resident (it's tiny: one
                            # value per (expert, output_dim)) before this
                            # module is replaced, same as the scale vectors.
                            gate_bias=getattr(child.gate_proj, "bias", None),
                            up_bias=getattr(child.up_proj, "bias", None),
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
    if swapped[0] == 0:
        raise ValueError(
            f"patch_model swapped 0 modules for {offload_dir} -- offload_index.json "
            "declares experts but no matching SwitchGLU was found in the model "
            "(fused switch layers like Laguna/MiniMax-M3-VL's gate_up_proj are not "
            "yet supported; see module docstring)."
        )
    store.swapped = swapped[0]
    return store


def _layer_id(path: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", path)
    return int(m.group(1)) if m else None


def _n_experts(switch_glu) -> int:
    """Number of experts in a SwitchGLU -- its ``gate_proj`` stacks per-expert
    weights as ``[E, out, in]`` (``[E, out, in//pack]`` when quantized), so
    axis 0 is the expert count. Falls back to 0 for an already-offloaded or
    otherwise nonstandard module (no ``gate_proj.weight``)."""
    w = getattr(getattr(switch_glu, "gate_proj", None), "weight", None)
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
