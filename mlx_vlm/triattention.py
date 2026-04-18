"""TriAttention: Trigonometric KV Cache Compression.

Implements the TriAttention method from "TriAttention: Efficient Long Reasoning
with Trigonometric KV Compression" (Lin et al., 2026). Scores key importance
using trigonometric series derived from pre-RoPE Q/K concentration, enabling
aggressive KV cache pruning with minimal accuracy loss.

Key insight: post-RoPE keys can be scored directly without inverse RoPE,
because the position-dependent terms cancel when combining calibration
Q-center phases with post-RoPE K phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import _BaseCache

# ──────────────────────────── defaults ────────────────────────────

DEFAULT_BUDGET = 2048
DEFAULT_DIVIDE_LENGTH = 128
DEFAULT_PROTECT_RECENT = 128
DEFAULT_PROTECT_INITIAL = 4
# Log-spaced future offsets: {1, 2, 4, ..., 2^16}
_DEFAULT_OFFSETS = mx.array([2**i for i in range(17)], dtype=mx.float32)


# ──────────────────────────── data classes ────────────────────────


@dataclass
class RoPEConfig:
    """RoPE configuration extracted from a model's attention layer."""

    head_dim: int  # full head dimension
    rotated_dims: int  # number of dimensions that are actually rotated
    traditional: bool  # True = half pairing, False = interleaved
    omega: mx.array  # [n_freqs] angular frequencies (rad/position)
    proportional: bool = False  # True for Gemma4 ProportionalRoPE layout


@dataclass
class TriAttentionCalibData:
    """Per-layer, per-head calibration statistics for TriAttention scoring.

    All arrays are indexed by layer and have shape [n_q_heads, n_freqs].
    The complex Q center is stored as (real, imag) pairs, and q_mean_norm
    stores the mean magnitude E[||q_f||].
    """

    q_center_real: Dict[int, mx.array]  # layer → [n_q_heads, n_freqs]
    q_center_imag: Dict[int, mx.array]  # layer → [n_q_heads, n_freqs]
    q_mean_norm: Dict[int, mx.array]  # layer → [n_q_heads, n_freqs]
    n_layers: int
    n_q_heads: int
    n_kv_heads: int


# ──────────────────────────── RoPE extraction ────────────────────


def extract_rope_config(model: nn.Module) -> Optional[RoPEConfig]:
    """Extract RoPE configuration from a language model.

    Inspects the first non-sliding attention layer's RoPE module to
    determine frequencies and rotation style. Prefers full-attention
    layers over sliding-window layers (important for Gemma4 where
    sliding layers use different RoPE parameters).

    Returns None for unsupported RoPE variants (MRoPE, xDRoPE, etc.).
    """
    layers = _find_layers(model)
    if layers is None or len(layers) == 0:
        return None

    # Try to find a full-attention layer first, fall back to layer 0
    target_layer = layers[0]
    for layer in layers:
        attn = _find_attention(layer)
        if attn is not None and not getattr(attn, "is_sliding", False):
            target_layer = layer
            break

    attn = _find_attention(target_layer)
    if attn is None:
        return None

    rope = getattr(attn, "rope", None)
    if rope is None:
        return None

    head_dim = _get_head_dim(attn)
    if head_dim is None:
        return None

    # Standard nn.RoPE
    if isinstance(rope, nn.RoPE):
        dims = rope.dims
        omega = _compute_omega_standard(dims, rope.base, rope.scale)
        return RoPEConfig(
            head_dim=head_dim,
            rotated_dims=dims,
            traditional=rope.traditional,
            omega=omega,
            proportional=False,
        )

    # ProportionalRoPE (Gemma4-style with partial rotation)
    if hasattr(rope, "_freqs") and hasattr(rope, "rotated_dims"):
        omega = 1.0 / rope._freqs
        return RoPEConfig(
            head_dim=head_dim,
            rotated_dims=rope.rotated_dims,
            traditional=rope.traditional,
            omega=omega,
            proportional=True,
        )

    return None


def extract_model_info(
    model: nn.Module,
) -> Optional[Tuple[int, int, int, int, RoPEConfig]]:
    """Extract (n_layers, n_q_heads, n_kv_heads, head_dim, rope_config).

    Uses the first non-sliding attention layer to determine head counts,
    matching the layer type that TriAttention will compress.
    """
    layers = _find_layers(model)
    if layers is None or len(layers) == 0:
        return None

    n_layers = len(layers)

    # Find the first non-sliding attention layer
    attn = None
    for layer in layers:
        candidate = _find_attention(layer)
        if candidate is not None and not getattr(candidate, "is_sliding", False):
            attn = candidate
            break
    if attn is None:
        attn = _find_attention(layers[0])
    if attn is None:
        return None

    n_q_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_heads", None)
    n_kv_heads = (
        getattr(attn, "n_kv_heads", None)
        or getattr(attn, "num_key_value_heads", None)
        or n_q_heads
    )
    head_dim = _get_head_dim(attn)

    if n_q_heads is None or head_dim is None:
        return None

    rope_config = extract_rope_config(model)
    if rope_config is None:
        return None

    return n_layers, n_q_heads, n_kv_heads, head_dim, rope_config


# ──────────────────────────── scoring ─────────────────────────────


def _decompose_complex(
    vectors: mx.array, config: RoPEConfig
) -> Tuple[mx.array, mx.array]:
    """Decompose vectors into (real, imag) per frequency band.

    Works for both pre-RoPE (calibration) and post-RoPE (cached keys)
    since RoPE rotates within pairs but preserves the layout.

    Args:
        vectors: [..., head_dim]
        config: RoPE configuration

    Returns:
        (real, imag): each [..., n_freqs]
    """
    n_freqs = config.rotated_dims // 2

    if config.proportional:
        # ProportionalRoPE: rotated portion is split across the two halves
        half = config.head_dim // 2
        rd_half = config.rotated_dims // 2
        portion = mx.concatenate(
            [vectors[..., :rd_half], vectors[..., half : half + rd_half]],
            axis=-1,
        )
        if config.traditional:
            real = portion[..., :n_freqs]
            imag = portion[..., n_freqs:]
        else:
            real = portion[..., 0::2]
            imag = portion[..., 1::2]
    else:
        if config.traditional:
            real = vectors[..., :n_freqs]
            imag = vectors[..., n_freqs : 2 * n_freqs]
        else:
            real = vectors[..., 0 : config.rotated_dims : 2]
            imag = vectors[..., 1 : config.rotated_dims : 2]

    return real, imag


def score_keys(
    cached_keys: mx.array,
    current_pos: int,
    calib: TriAttentionCalibData,
    layer_idx: int,
    rope_config: RoPEConfig,
    offsets: mx.array = _DEFAULT_OFFSETS,
) -> mx.array:
    """Score cached keys for importance using the trigonometric series.

    Uses post-RoPE keys directly — no inverse RoPE needed, because
    position-dependent terms cancel in the phase difference.

    The score for each key is:
        S(k) = (1/|D|) Σ_δ S_trig(k, t_δ) + S_norm(k)
    where t_δ = current_pos + δ.

    For GQA: z-score normalize per query head, then take max.

    Args:
        cached_keys: [B, n_kv_heads, S, head_dim] post-RoPE keys
        current_pos: absolute position of the current token
        calib: calibration data with Q centers
        layer_idx: transformer layer index
        rope_config: RoPE configuration
        offsets: [n_offsets] future position offsets

    Returns:
        [B, n_kv_heads, S] importance score per key
    """
    B, H_kv, S, _ = cached_keys.shape

    # 1. Extract magnitude and phase from post-RoPE keys
    k_real, k_imag = _decompose_complex(cached_keys, rope_config)
    k_mag = mx.sqrt(k_real * k_real + k_imag * k_imag + 1e-12)  # [B,H,S,F]
    k_phase = mx.arctan2(k_imag, k_real)  # [B,H,S,F]

    # 2. Load calibration Q centers for this layer
    q_cr = calib.q_center_real[layer_idx]  # [Q_total, F]
    q_ci = calib.q_center_imag[layer_idx]
    q_mn = calib.q_mean_norm[layer_idx]

    q_center_mag = mx.sqrt(q_cr * q_cr + q_ci * q_ci + 1e-12)  # [Q, F]
    q_center_phase = mx.arctan2(q_ci, q_cr)  # [Q, F]

    # 3. Reshape for GQA: [H_kv, G, F] where G = queries per KV head
    G = calib.n_q_heads // calib.n_kv_heads
    n_freqs = rope_config.rotated_dims // 2
    q_center_mag = q_center_mag.reshape(H_kv, G, n_freqs)
    q_center_phase = q_center_phase.reshape(H_kv, G, n_freqs)
    q_mean_norm = q_mn.reshape(H_kv, G, n_freqs)

    omega = rope_config.omega  # [F]

    # 4. Phase difference: φ = arg(q̄) - arg(k_rot)
    #    Shapes: [1,H,1,G,F] - [B,H,S,1,F] → [B,H,S,G,F]
    phi = q_center_phase[None, :, None, :, :] - k_phase[:, :, :, None, :]

    # 5. Amplitude: |q̄| · |k_rot|
    amp = q_center_mag[None, :, None, :, :] * k_mag[:, :, :, None, :]

    # 6. Efficient trig scoring via a·cos_tw - b·sin_tw decomposition
    #    score(t) = Σ_f amp_f · cos(ω_f·t + φ_f)
    #             = Σ_f [a_f · cos(ω_f·t)] - Σ_f [b_f · sin(ω_f·t)]
    #    where a_f = amp_f·cos(φ_f), b_f = amp_f·sin(φ_f)
    a = amp * mx.cos(phi)  # [B,H,S,G,F]
    b = amp * mx.sin(phi)  # [B,H,S,G,F]

    # Precompute cos/sin tables for all offsets
    t = (current_pos + offsets).astype(mx.float32)  # [n_offsets]
    t_omega = t[:, None] * omega[None, :]  # [n_offsets, F]
    cos_tw = mx.cos(t_omega)  # [n_offsets, F]
    sin_tw = mx.sin(t_omega)  # [n_offsets, F]

    # Matrix multiply: [B*H*S*G, F] @ [F, n_offsets] → [B*H*S*G, n_offsets]
    flat_shape = (B * H_kv * S * G, n_freqs)
    s_trig_flat = a.reshape(flat_shape) @ cos_tw.T - b.reshape(flat_shape) @ sin_tw.T

    # Average over offsets → [B,H,S,G]
    s_trig = mx.mean(s_trig_flat, axis=-1).reshape(B, H_kv, S, G)

    # 7. S_norm (position-independent): Σ_f (E[||q_f||] - ||E[q_f]||) · ||k_f||
    norm_weight = q_mean_norm - q_center_mag  # [H,G,F]
    s_norm = mx.sum(
        norm_weight[None, :, None, :, :] * k_mag[:, :, :, None, :],
        axis=-1,
    )  # [B,H,S,G]

    # 8. Combined score
    s = s_trig + s_norm  # [B,H,S,G]

    # 9. GQA aggregation: z-score per query head, max across group
    if G > 1:
        mean_s = mx.mean(s, axis=2, keepdims=True)  # [B,H,1,G]
        var_s = mx.mean((s - mean_s) ** 2, axis=2, keepdims=True)
        z = (s - mean_s) / mx.sqrt(var_s + 1e-8)
        scores = mx.max(z, axis=-1)  # [B,H,S]
    else:
        scores = s.squeeze(-1)  # [B,H,S]

    return scores


# ──────────────────────────── KV cache ────────────────────────────


class TriAttentionKVCache(_BaseCache):
    """KV cache with trigonometric-series-based compression.

    Drop-in replacement for KVCache. When the cache exceeds ``budget``,
    scores all keys via the TriAttention trigonometric series and retains
    only the top-scoring ones. Compression triggers every ``divide_length``
    generated tokens.

    Attention sinks (first ``protect_initial`` tokens) and recent context
    (last ``protect_recent`` tokens) are always retained.
    """

    def __init__(
        self,
        budget: int = DEFAULT_BUDGET,
        calib: Optional[TriAttentionCalibData] = None,
        layer_idx: int = 0,
        rope_config: Optional[RoPEConfig] = None,
        divide_length: int = DEFAULT_DIVIDE_LENGTH,
        protect_recent: int = DEFAULT_PROTECT_RECENT,
        protect_initial: int = DEFAULT_PROTECT_INITIAL,
    ):
        self.budget = budget
        self.calib = calib
        self.layer_idx = layer_idx
        self.rope_config = rope_config
        self.divide_length = divide_length
        self.protect_recent = protect_recent
        self.protect_initial = protect_initial

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0
        self._tokens_since_compress: int = 0
        self._offsets = _DEFAULT_OFFSETS

    @classmethod
    def from_cache(
        cls,
        cache: Any,
        budget: int,
        calib: TriAttentionCalibData,
        layer_idx: int,
        rope_config: RoPEConfig,
        **kwargs,
    ) -> "TriAttentionKVCache":
        """Hot-swap from an existing KVCache."""
        inst = cls(
            budget=budget,
            calib=calib,
            layer_idx=layer_idx,
            rope_config=rope_config,
            **kwargs,
        )
        keys, values = cache.state
        if keys is not None:
            inst.keys = keys
            inst.values = values
            inst.offset = cache.offset
            inst._tokens_since_compress = cache.offset
        return inst

    @property
    def _physical_size(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        n_new = keys.shape[2]

        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset += n_new
        self._tokens_since_compress += n_new

        # Trigger compression when cache exceeds budget and enough new tokens
        if (
            self._physical_size > self.budget
            and self._tokens_since_compress >= self.divide_length
            and self.calib is not None
            and self.rope_config is not None
        ):
            self._compress()

        return self.keys, self.values

    def _compress(self):
        S = self._physical_size
        if S <= self.budget:
            return

        # Score all keys
        scores = score_keys(
            self.keys,
            self.offset,
            self.calib,
            self.layer_idx,
            self.rope_config,
            self._offsets,
        )  # [B, n_kv_heads, S]

        # Average across KV heads for unified token selection
        avg_scores = mx.mean(scores, axis=1)  # [B, S]

        # Protect attention sinks and recent tokens
        if self.protect_initial > 0:
            avg_scores = mx.concatenate(
                [
                    mx.full(
                        (avg_scores.shape[0], self.protect_initial),
                        1e9,
                        dtype=avg_scores.dtype,
                    ),
                    avg_scores[:, self.protect_initial :],
                ],
                axis=1,
            )
        if self.protect_recent > 0 and S > self.protect_recent:
            avg_scores = mx.concatenate(
                [
                    avg_scores[:, : -self.protect_recent],
                    mx.full(
                        (avg_scores.shape[0], self.protect_recent),
                        1e9,
                        dtype=avg_scores.dtype,
                    ),
                ],
                axis=1,
            )

        # Select top-budget tokens
        keep_count = min(self.budget, S)
        keep_idx = mx.argpartition(-avg_scores[0], kth=keep_count - 1)[:keep_count]
        keep_idx = mx.sort(keep_idx)  # preserve temporal order

        # Gather retained keys/values
        self.keys = self.keys[:, :, keep_idx, :]
        self.values = self.values[:, :, keep_idx, :]
        self._tokens_since_compress = 0

        mx.eval(self.keys, self.values)

    @property
    def state(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v

    @property
    def nbytes(self) -> int:
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes
        if self.values is not None:
            total += self.values.nbytes
        return total

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._physical_size, n)
        if n > 0 and self.keys is not None:
            self.keys = self.keys[:, :, n:, :]
            self.values = self.values[:, :, n:, :]
        self.offset -= n
        return n

    @property
    def meta_state(self):
        return tuple(map(str, (self.budget, self.offset, self._tokens_since_compress)))

    @meta_state.setter
    def meta_state(self, v):
        self.budget, self.offset, self._tokens_since_compress = map(int, v)


# ──────────────────────────── calibration I/O ─────────────────────


def save_calibration(calib: TriAttentionCalibData, path: str) -> None:
    """Save calibration data to safetensors file."""
    import numpy as np
    from safetensors.numpy import save_file

    data = {}
    for layer_idx in range(calib.n_layers):
        # Convert to float32 for safetensors compatibility (bfloat16 not supported)
        data[f"layer.{layer_idx}.q_center_real"] = np.array(
            calib.q_center_real[layer_idx].astype(mx.float32)
        )
        data[f"layer.{layer_idx}.q_center_imag"] = np.array(
            calib.q_center_imag[layer_idx].astype(mx.float32)
        )
        data[f"layer.{layer_idx}.q_mean_norm"] = np.array(
            calib.q_mean_norm[layer_idx].astype(mx.float32)
        )

    metadata = {
        "n_layers": str(calib.n_layers),
        "n_q_heads": str(calib.n_q_heads),
        "n_kv_heads": str(calib.n_kv_heads),
    }

    save_file(data, path, metadata=metadata)


def load_calibration(path: str) -> TriAttentionCalibData:
    """Load calibration data from safetensors file."""
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensors[key] = mx.array(f.get_tensor(key))

    n_layers = int(metadata["n_layers"])
    n_q_heads = int(metadata["n_q_heads"])
    n_kv_heads = int(metadata["n_kv_heads"])

    q_center_real = {}
    q_center_imag = {}
    q_mean_norm = {}

    for i in range(n_layers):
        q_center_real[i] = tensors[f"layer.{i}.q_center_real"]
        q_center_imag[i] = tensors[f"layer.{i}.q_center_imag"]
        q_mean_norm[i] = tensors[f"layer.{i}.q_mean_norm"]

    return TriAttentionCalibData(
        q_center_real=q_center_real,
        q_center_imag=q_center_imag,
        q_mean_norm=q_mean_norm,
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
    )


# ──────────────────────────── generation integration ──────────────


def maybe_apply_triattention(
    prompt_cache: List[Any],
    model: nn.Module,
    calib_path: str,
    budget: int = DEFAULT_BUDGET,
    divide_length: int = DEFAULT_DIVIDE_LENGTH,
    protect_recent: int = DEFAULT_PROTECT_RECENT,
    protect_initial: int = DEFAULT_PROTECT_INITIAL,
) -> None:
    """Convert standard KVCache entries to TriAttentionKVCache in-place.

    Follows the same pattern as maybe_quantize_kv_cache for TurboQuant.
    """
    from .models.cache import CacheList, KVCache, RotatingKVCache

    calib = load_calibration(calib_path)
    rope_config = extract_rope_config(model)
    if rope_config is None:
        raise ValueError(
            "TriAttention: could not extract RoPE config from model. "
            "This model may use an unsupported RoPE variant (MRoPE, xDRoPE, etc.)."
        )

    def convert_entry(entry, layer_idx):
        if isinstance(entry, TriAttentionKVCache):
            return entry
        if isinstance(entry, RotatingKVCache):
            return entry  # Don't wrap sliding window caches
        if isinstance(entry, KVCache):
            if entry.offset == 0:
                return TriAttentionKVCache(
                    budget=budget,
                    calib=calib,
                    layer_idx=layer_idx,
                    rope_config=rope_config,
                    divide_length=divide_length,
                    protect_recent=protect_recent,
                    protect_initial=protect_initial,
                )
            return TriAttentionKVCache.from_cache(
                entry,
                budget=budget,
                calib=calib,
                layer_idx=layer_idx,
                rope_config=rope_config,
                divide_length=divide_length,
                protect_recent=protect_recent,
                protect_initial=protect_initial,
            )
        if isinstance(entry, CacheList):
            entry.caches = [convert_entry(sub, layer_idx) for sub in entry.caches]
            return entry
        if isinstance(entry, list):
            for i, sub in enumerate(entry):
                entry[i] = convert_entry(sub, layer_idx)
            return entry
        return entry

    for layer_idx in range(len(prompt_cache)):
        prompt_cache[layer_idx] = convert_entry(prompt_cache[layer_idx], layer_idx)


# ──────────────────────────── online calibration ──────────────────


class _OnlineCaptureWrapper:
    """Lightweight wrapper that captures pre-RoPE Q during prefill.

    Same approach as the calibration script's _CaptureWrapper, but designed
    for transient use during a single generation call.
    """

    def __init__(self, wrapped: nn.Module, capture_list: list):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_capture_list", capture_list)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __call__(self, x, mask=None, cache=None, **kwargs):
        wrapped = object.__getattribute__(self, "_wrapped")
        capture_list = object.__getattribute__(self, "_capture_list")

        B, L, _ = x.shape
        n_heads = getattr(wrapped, "n_heads", None) or getattr(
            wrapped, "num_heads", None
        )
        if n_heads is not None:
            q = wrapped.q_proj(x).reshape(B, L, n_heads, -1)
            if hasattr(wrapped, "q_norm"):
                q = wrapped.q_norm(q)
            capture_list.append(mx.stop_gradient(q))

        return wrapped(x, mask=mask, cache=cache, **kwargs)


class OnlineCalibrationState:
    """Holds hooks and captures for online TriAttention calibration.

    Usage in generate_step::

        # Before prefill
        online_state = setup_online_triattention(model, budget=512)

        # ... prefill runs, hooks capture Q vectors ...

        # After prefill, before decode loop
        activate_online_triattention(online_state, prompt_cache)
    """

    def __init__(self):
        self.hooks: list = []  # (layer, attr_name, original_attn)
        self.captures: Dict[int, list] = {}
        self.budget: int = DEFAULT_BUDGET
        self.divide_length: int = DEFAULT_DIVIDE_LENGTH
        self.protect_recent: int = DEFAULT_PROTECT_RECENT
        self.protect_initial: int = DEFAULT_PROTECT_INITIAL
        self.rope_config: Optional[RoPEConfig] = None
        self.model_info: Optional[tuple] = None


def setup_online_triattention(
    model: nn.Module,
    budget: int = DEFAULT_BUDGET,
    divide_length: int = DEFAULT_DIVIDE_LENGTH,
    protect_recent: int = DEFAULT_PROTECT_RECENT,
    protect_initial: int = DEFAULT_PROTECT_INITIAL,
) -> OnlineCalibrationState:
    """Install capture hooks for online calibration. Call before prefill.

    Hooks capture pre-RoPE Q vectors from all full-attention layers during
    the prefill phase. After prefill, call :func:`activate_online_triattention`
    to compute calibration and convert caches.
    """
    state = OnlineCalibrationState()
    state.budget = budget
    state.divide_length = divide_length
    state.protect_recent = protect_recent
    state.protect_initial = protect_initial

    info = extract_model_info(model)
    if info is None:
        raise ValueError(
            "TriAttention: could not extract model info. " "Unsupported architecture."
        )
    state.model_info = info
    state.rope_config = info[4]

    # Find the language model for hooking
    lm = model
    if hasattr(model, "language_model"):
        lm_prop = model.language_model
        if lm_prop is not model:
            lm = lm_prop

    layers = _find_layers(lm)
    if layers is None:
        raise ValueError("Cannot find transformer layers")

    for layer_idx, layer in enumerate(layers):
        attr_name = None
        attn = None
        for name in ("self_attn", "attention"):
            if hasattr(layer, name):
                attr_name = name
                attn = getattr(layer, name)
                break
        if attn is None:
            continue

        # Skip sliding-window layers
        if getattr(attn, "is_sliding", False):
            continue

        state.captures[layer_idx] = []
        wrapper = _OnlineCaptureWrapper(attn, state.captures[layer_idx])
        setattr(layer, attr_name, wrapper)
        state.hooks.append((layer, attr_name, attn))

    return state


def activate_online_triattention(
    state: OnlineCalibrationState,
    prompt_cache: List[Any],
) -> None:
    """Compute calibration from captured Q vectors and activate compression.

    Call after prefill completes. Removes capture hooks, computes Q-center
    statistics from the prefill tokens, and converts KVCache entries to
    TriAttentionKVCache.
    """
    from .models.cache import CacheList, KVCache, RotatingKVCache

    # 1. Remove hooks
    for layer, attr_name, original_attn in state.hooks:
        setattr(layer, attr_name, original_attn)

    n_layers, n_q_heads, n_kv_heads, head_dim, rope_config = state.model_info
    n_freqs = rope_config.rotated_dims // 2

    # 2. Compute calibration from captures
    q_center_real = {}
    q_center_imag = {}
    q_mean_norm = {}

    for layer_idx in range(n_layers):
        if layer_idx not in state.captures or not state.captures.get(layer_idx):
            q_center_real[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            q_center_imag[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            q_mean_norm[layer_idx] = mx.zeros((n_q_heads, n_freqs))
            continue

        all_q = mx.concatenate(state.captures[layer_idx], axis=1)
        mx.eval(all_q)

        cr_list, ci_list, mn_list = [], [], []
        for h in range(n_q_heads):
            q_head = all_q[0, :, h, :]
            real, imag = _decompose_complex(q_head, rope_config)
            cr_list.append(mx.mean(real, axis=0))
            ci_list.append(mx.mean(imag, axis=0))
            mag = mx.sqrt(real * real + imag * imag + 1e-12)
            mn_list.append(mx.mean(mag, axis=0))

        q_center_real[layer_idx] = mx.stack(cr_list)
        q_center_imag[layer_idx] = mx.stack(ci_list)
        q_mean_norm[layer_idx] = mx.stack(mn_list)
        mx.eval(
            q_center_real[layer_idx],
            q_center_imag[layer_idx],
            q_mean_norm[layer_idx],
        )

    calib = TriAttentionCalibData(
        q_center_real=q_center_real,
        q_center_imag=q_center_imag,
        q_mean_norm=q_mean_norm,
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
    )

    # 3. Free capture memory
    state.captures.clear()
    state.hooks.clear()

    # 4. Convert caches to TriAttentionKVCache
    def convert_entry(entry, layer_idx):
        if isinstance(entry, TriAttentionKVCache):
            return entry
        if isinstance(entry, RotatingKVCache):
            return entry
        if isinstance(entry, KVCache):
            return TriAttentionKVCache.from_cache(
                entry,
                budget=state.budget,
                calib=calib,
                layer_idx=layer_idx,
                rope_config=rope_config,
                divide_length=state.divide_length,
                protect_recent=state.protect_recent,
                protect_initial=state.protect_initial,
            )
        if isinstance(entry, CacheList):
            entry.caches = [convert_entry(sub, layer_idx) for sub in entry.caches]
            return entry
        if isinstance(entry, list):
            for i, sub in enumerate(entry):
                entry[i] = convert_entry(sub, layer_idx)
            return entry
        return entry

    for layer_idx in range(len(prompt_cache)):
        prompt_cache[layer_idx] = convert_entry(prompt_cache[layer_idx], layer_idx)


# ──────────────────────────── private helpers ─────────────────────


def _find_layers(model: nn.Module) -> Optional[list]:
    """Find the transformer layer list from a model."""
    lm = model
    if hasattr(model, "language_model"):
        lm_prop = model.language_model
        # Handle property that returns self
        if lm_prop is not model:
            lm = lm_prop
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    return None


def _find_attention(layer: nn.Module) -> Optional[nn.Module]:
    """Find the attention sub-module in a transformer layer."""
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)


def _get_head_dim(attn: nn.Module) -> Optional[int]:
    """Get head_dim from an attention module."""
    hd = getattr(attn, "head_dim", None)
    if hd is not None:
        return hd
    n_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_heads", None)
    if n_heads and hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
        return attn.q_proj.weight.shape[0] // n_heads
    return None


def _compute_omega_standard(dims: int, base: float, scale: float) -> mx.array:
    """Compute angular frequencies for standard nn.RoPE."""
    exponents = mx.arange(0, dims, 2, dtype=mx.float32) / dims
    return (1.0 / (base**exponents)) / scale
