"""AWQ (activation-aware weight quantization).

Rescales weights so channels with large activations are protected from
quantization error, folding the inverse scale into the preceding norm or linear
so the full-precision model is unchanged, then optionally clips each group to
shrink the remaining round-to-nearest error. Run before
``quant_utils.quantize_model``. Pure mlx.core/mlx.nn.

Scaling groups are discovered generically (attention and MLP in standard and
gated-delta decoder blocks, plus vision transformer blocks), so hybrid and
vision modules are covered rather than left to RTN.

Reference: "AWQ: Activation-aware Weight Quantization" (arXiv 2306.00978).
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _fake_quant(w: mx.array, group_size: int, bits: int) -> mx.array:
    wq, scales, biases = mx.quantize(w, group_size, bits)
    return mx.dequantize(wq, scales, biases, group_size, bits)


def _divisible(linears: List[nn.Module], group_size: int) -> bool:
    return all(int(l.weight.shape[-1]) % group_size == 0 for l in linears)


def _is_norm(module) -> bool:
    return isinstance(module, (nn.RMSNorm, nn.LayerNorm))


def _search_scale(
    weights: List[mx.array],
    inputs: mx.array,
    act_scale: mx.array,
    group_size: int,
    bits: int,
    n_grid: int,
    max_out: int = 256,
) -> mx.array:
    """Grid-search the per-input-channel AWQ scale minimizing output MSE.

    The reconstruction error is estimated on a strided subsample of at most
    ``max_out`` output channels per weight, which keeps the per-grid-point
    matmul cheap for large layers without changing the scale that is applied.
    """
    x = inputs.astype(mx.float32)
    wcat = mx.concatenate([w.astype(mx.float32) for w in weights], axis=0)
    w_scale = mx.mean(mx.abs(wcat), axis=0) + 1e-6

    subs = []
    for w in weights:
        wf = w.astype(mx.float32)
        if wf.shape[0] > max_out:
            step = wf.shape[0] // max_out
            wf = wf[::step][:max_out]
        subs.append(wf)
    refs = [wf @ x.T for wf in subs]

    best_err: Optional[float] = None
    best_s = mx.ones_like(act_scale)
    for g in range(n_grid):
        ratio = g / max(n_grid - 1, 1)
        s = (act_scale**ratio) / (w_scale ** (1.0 - ratio))
        s = s / mx.sqrt(mx.max(s) * mx.min(s))
        s = mx.clip(s, 1e-4, 1e4)
        xs = x / s
        err = mx.array(0.0)
        for wf, ref in zip(subs, refs):
            wq = _fake_quant(wf * s, group_size, bits)
            err = err + mx.sum((wq @ xs.T - ref) ** 2)
        mx.eval(err)
        e = float(err.item())
        if best_err is None or e < best_err:
            best_err = e
            best_s = s
    mx.eval(best_s)
    return best_s


def _search_clip(
    weight: mx.array,
    inputs: mx.array,
    group_size: int,
    bits: int,
    n_grid: int = 10,
    min_ratio: float = 0.5,
    max_out: int = 256,
) -> mx.array:
    """Return ``weight`` with its quantization groups clamped to the range that
    minimizes the *activation-aware* reconstruction error ``||q(clip(w))x - wx||``.

    Shrinking the range resolves the bulk of the weights more finely; the error
    is measured through the calibration activations (on a strided subsample of at
    most ``max_out`` output rows) so a value is only clamped when doing so costs
    little at the output -- clamping in weight space alone would never remove a
    large but output-critical outlier.
    """
    w = weight.astype(mx.float32)
    x = inputs.astype(mx.float32)
    out, in_dim = int(w.shape[0]), int(w.shape[-1])
    n_groups = in_dim // group_size

    wr = w
    if out > max_out:
        wr = w[:: out // max_out][:max_out]
    ref = wr @ x.T
    wrg = wr.reshape(wr.shape[0], n_groups, group_size)
    max_abs = mx.max(mx.abs(wrg), axis=-1, keepdims=True)

    best_ratio = 1.0
    best_err: Optional[float] = None
    for i in range(n_grid):
        ratio = min_ratio + (1.0 - min_ratio) * i / max(n_grid - 1, 1)
        cw = mx.clip(wrg, -ratio * max_abs, ratio * max_abs).reshape(
            wr.shape[0], in_dim
        )
        err = mx.sum((_fake_quant(cw, group_size, bits) @ x.T - ref) ** 2)
        mx.eval(err)
        e = float(err.item())
        if best_err is None or e < best_err:
            best_err = e
            best_ratio = ratio

    wg = w.reshape(out, n_groups, group_size)
    full_max = mx.max(mx.abs(wg), axis=-1, keepdims=True)
    clipped = mx.clip(wg, -best_ratio * full_max, best_ratio * full_max)
    return clipped.reshape(out, in_dim).astype(weight.dtype)


def _fold_into_norm(norm: nn.Module, linears: List[nn.Module], s: mx.array) -> None:
    inv = 1.0 / s
    norm.weight = (norm.weight * inv).astype(norm.weight.dtype)
    if "bias" in norm:  # LayerNorm carries a bias that shares the scale
        norm.bias = (norm.bias * inv).astype(norm.bias.dtype)
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


def _source_out_dim(source: nn.Module) -> int:
    # norm: last axis; linear: output rows
    if _is_norm(source):
        return int(source.weight.shape[-1])
    return int(source.weight.shape[0])


def _group_ok(source: nn.Module, linears: List[nn.Module]) -> bool:
    if source is None or not linears:
        return False
    dim = _source_out_dim(source)
    return all(int(l.weight.shape[-1]) == dim for l in linears)


def _linear_children(module, names) -> List[nn.Module]:
    out = []
    for name in names:
        child = getattr(module, name, None)
        if isinstance(child, nn.Linear):
            out.append(child)
    return out


def _awq_groups(layer) -> List[Tuple[nn.Module, List[nn.Module]]]:
    """Discover ``(source, targets)`` scaling groups in a decoder or vision block.

    ``source`` is a norm (inverse scale folds into its weight/bias) or a linear
    (folds into its output rows). Covers standard and gated-delta attention,
    the SwiGLU MLP, and vision-transformer blocks; unknown structures yield no
    groups and are left to RTN.
    """
    groups: List[Tuple[nn.Module, List[nn.Module]]] = []

    # --- decoder block ---
    in_norm = getattr(layer, "input_layernorm", None)
    post_norm = getattr(layer, "post_attention_layernorm", None)
    attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
    if attn is not None and _is_norm(in_norm):
        targets = _linear_children(
            attn,
            ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z"),
        )
        if targets:
            groups.append((in_norm, targets))
    if attn is not None:
        v = getattr(attn, "v_proj", None)
        o = getattr(attn, "o_proj", None)
        if isinstance(v, nn.Linear) and isinstance(o, nn.Linear):
            groups.append((v, [o]))

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        gate_up = _linear_children(mlp, ("gate_proj", "up_proj"))
        if gate_up and _is_norm(post_norm):
            groups.append((post_norm, gate_up))
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if isinstance(up, nn.Linear) and isinstance(down, nn.Linear):
            groups.append((up, [down]))

    # --- vision transformer block (LayerNorm, fused qkv, fc1/fc2) ---
    norm1 = getattr(layer, "norm1", None)
    norm2 = getattr(layer, "norm2", None)
    vattn = getattr(layer, "attn", None)
    if _is_norm(norm1) and vattn is not None:
        qkv = _linear_children(vattn, ("qkv",))
        if qkv:
            groups.append((norm1, qkv))
    if _is_norm(norm2) and mlp is not None:
        fc1 = getattr(mlp, "linear_fc1", None) or getattr(mlp, "fc1", None)
        fc2 = getattr(mlp, "linear_fc2", None) or getattr(mlp, "fc2", None)
        if isinstance(fc1, nn.Linear):
            groups.append((norm2, [fc1]))
        if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
            groups.append((fc1, [fc2]))

    return groups


def _apply_group(
    source, targets, id2stats, group_size, bits, n_grid, clip, clip_grid
) -> bool:
    st = id2stats.get(id(targets[0]))
    if st is None or st.get("inputs") is None:
        return False
    if not _group_ok(source, targets) or not _divisible(targets, group_size):
        return False

    s = _search_scale(
        [t.weight for t in targets], st["inputs"], st["scale"], group_size, bits, n_grid
    )
    if isinstance(source, nn.Linear):
        _fold_into_linear(source, targets, s)
    else:
        _fold_into_norm(source, targets, s)

    if clip:
        xs = st["inputs"].astype(mx.float32) / s
        for t in targets:
            t.weight = _search_clip(t.weight, xs, group_size, bits, n_grid=clip_grid)
        mx.eval([t.weight for t in targets])
    return True


def apply_awq(
    model: nn.Module,
    stats: Dict[str, dict],
    bits: int = 4,
    group_size: int = 64,
    n_grid: int = 20,
    clip: bool = True,
    clip_grid: int = 10,
) -> Dict[str, int]:
    """Apply AWQ scaling (and optional clipping) to every discoverable block.

    ``stats`` is the output of :func:`collect_activation_stats`. Groups whose
    shared input was not seen during calibration, or whose structure is not
    recognised, are left untouched (RTN handles them). Returns a summary
    ``{"blocks": .., "groups": ..}``.
    """
    id2stats: Dict[int, dict] = {}
    for path, module in model.named_modules():
        if path in stats:
            id2stats[id(module)] = stats[path]

    blocks_done = 0
    groups_done = 0
    for _, module in model.named_modules():
        groups = _awq_groups(module)
        if not groups:
            continue
        applied = 0
        for source, targets in groups:
            if _apply_group(
                source, targets, id2stats, group_size, bits, n_grid, clip, clip_grid
            ):
                applied += 1
        if applied:
            blocks_done += 1
            groups_done += applied
    return {"blocks": blocks_done, "groups": groups_done}
