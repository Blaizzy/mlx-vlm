"""In-house AWQ (activation-aware weight quantization).

Rescales decoder-layer weights so that channels with large activations are
protected from quantization error, folding the inverse scale into the
preceding RMSNorm or linear so the full-precision model is unchanged. Run
before ``quant_utils.quantize_model``. Pure mlx.core/mlx.nn.

Reference: "AWQ: Activation-aware Weight Quantization" (arXiv 2306.00978).
"""

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn


def _fake_quant(w: mx.array, group_size: int, bits: int) -> mx.array:
    wq, scales, biases = mx.quantize(w, group_size, bits)
    return mx.dequantize(wq, scales, biases, group_size, bits)


def _divisible(linears: List[nn.Module], group_size: int) -> bool:
    return all(int(l.weight.shape[-1]) % group_size == 0 for l in linears)


def _search_scale(
    weights: List[mx.array],
    inputs: mx.array,
    act_scale: mx.array,
    group_size: int,
    bits: int,
    n_grid: int,
) -> mx.array:
    x = inputs.astype(mx.float32)
    wcat = mx.concatenate([w.astype(mx.float32) for w in weights], axis=0)
    w_scale = mx.mean(mx.abs(wcat), axis=0) + 1e-6
    refs = [w.astype(mx.float32) @ x.T for w in weights]

    best_err: Optional[float] = None
    best_s = mx.ones_like(act_scale)
    for g in range(n_grid):
        ratio = g / max(n_grid - 1, 1)
        s = (act_scale**ratio) / (w_scale ** (1.0 - ratio))
        s = s / mx.sqrt(mx.max(s) * mx.min(s))
        s = mx.clip(s, 1e-4, 1e4)
        xs = x / s
        err = mx.array(0.0)
        for w, ref in zip(weights, refs):
            wq = _fake_quant(w.astype(mx.float32) * s, group_size, bits)
            err = err + mx.sum((wq @ xs.T - ref) ** 2)
        mx.eval(err)
        e = float(err.item())
        if best_err is None or e < best_err:
            best_err = e
            best_s = s
    mx.eval(best_s)
    return best_s


def _fold_into_norm(norm: nn.Module, linears: List[nn.Module], s: mx.array) -> None:
    inv = 1.0 / s
    norm.weight = (norm.weight * inv).astype(norm.weight.dtype)
    for lin in linears:
        lin.weight = (lin.weight * s).astype(lin.weight.dtype)
    mx.eval([norm.weight] + [lin.weight for lin in linears])


def _fold_into_linear(prev: nn.Module, linears: List[nn.Module], s: mx.array) -> None:
    inv = 1.0 / s
    prev.weight = (prev.weight * inv[:, None]).astype(prev.weight.dtype)
    if "bias" in prev:
        prev.bias = (prev.bias * inv).astype(prev.bias.dtype)
    for lin in linears:
        lin.weight = (lin.weight * s).astype(lin.weight.dtype)
    mx.eval([prev.weight] + [lin.weight for lin in linears])


def _norm_ok(norm: nn.Module, linears: List[nn.Module]) -> bool:
    if type(norm) is not nn.RMSNorm:
        return False
    return all(int(norm.weight.shape[-1]) == int(l.weight.shape[-1]) for l in linears)


def _awq_layer(layer, id2stats, group_size, bits, n_grid) -> int:
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    in_norm = getattr(layer, "input_layernorm", None)
    post_norm = getattr(layer, "post_attention_layernorm", None)
    if attn is None or mlp is None:
        return 0

    q = getattr(attn, "q_proj", None)
    k = getattr(attn, "k_proj", None)
    v = getattr(attn, "v_proj", None)
    o = getattr(attn, "o_proj", None)
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    parts = (q, k, v, o, gate, up, down)
    if any(p is None for p in parts) or not all(
        isinstance(p, nn.Linear) for p in parts
    ):
        return 0

    done = 0

    st = id2stats.get(id(q))
    if (
        st is not None
        and st["inputs"] is not None
        and _norm_ok(in_norm, [q, k, v])
        and _divisible([q, k, v], group_size)
    ):
        s = _search_scale(
            [q.weight, k.weight, v.weight],
            st["inputs"],
            st["scale"],
            group_size,
            bits,
            n_grid,
        )
        _fold_into_norm(in_norm, [q, k, v], s)
        done += 1

    st = id2stats.get(id(gate))
    if (
        st is not None
        and st["inputs"] is not None
        and _norm_ok(post_norm, [gate, up])
        and _divisible([gate, up], group_size)
    ):
        s = _search_scale(
            [gate.weight, up.weight],
            st["inputs"],
            st["scale"],
            group_size,
            bits,
            n_grid,
        )
        _fold_into_norm(post_norm, [gate, up], s)
        done += 1

    st = id2stats.get(id(down))
    if (
        st is not None
        and st["inputs"] is not None
        and int(up.weight.shape[0]) == int(down.weight.shape[-1])
        and _divisible([down], group_size)
    ):
        s = _search_scale(
            [down.weight], st["inputs"], st["scale"], group_size, bits, n_grid
        )
        _fold_into_linear(up, [down], s)
        done += 1

    st = id2stats.get(id(o))
    if (
        st is not None
        and st["inputs"] is not None
        and int(v.weight.shape[0]) == int(o.weight.shape[-1])
        and _divisible([o], group_size)
    ):
        s = _search_scale(
            [o.weight], st["inputs"], st["scale"], group_size, bits, n_grid
        )
        _fold_into_linear(v, [o], s)
        done += 1

    return done


def apply_awq(
    model: nn.Module,
    stats: Dict[str, dict],
    bits: int = 4,
    group_size: int = 64,
    n_grid: int = 20,
) -> Dict[str, int]:
    """Apply AWQ scaling to every standard decoder layer reachable in ``model``.

    ``stats`` is the output of :func:`collect_activation_stats`. Returns a
    summary ``{"layers": .., "groups": ..}``. Layers that do not match the
    standard attention/MLP structure are left untouched (RTN handles them).
    """
    id2stats: Dict[int, dict] = {}
    for path, module in model.named_modules():
        if path in stats:
            id2stats[id(module)] = stats[path]

    layers_done = 0
    groups_done = 0
    for _, module in model.named_modules():
        if (
            getattr(module, "self_attn", None) is not None
            and getattr(module, "mlp", None) is not None
        ):
            g = _awq_layer(module, id2stats, group_size, bits, n_grid)
            if g:
                layers_done += 1
                groups_done += g
    return {"layers": layers_done, "groups": groups_done}
