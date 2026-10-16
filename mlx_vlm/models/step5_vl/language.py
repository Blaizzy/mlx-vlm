"""Step 5: windowed GQA, region-selected attention and FP32 residuals.

The sparse indexer follows StepFun's published inference/model.py and kernel.py.
Each KV group selects completed eight-token regions plus the current region.
"""

import mlx.core as mx
import mlx.nn as nn

from ..base import create_attention_mask
from ..cache import CacheList, KVCache, RotatingKVCache
from ..step3p7.config import _remap_language_key
from ..step3p7.language import MLP
from ..step3p7.language import Attention as StepAttention
from ..step3p7.language import LanguageModel as StepLanguageModel
from ..step3p7.language import StepRMSNorm
from ..switch_layers import SwitchGLU


def round_e4m3(x):
    """E4M3 nearest-even rounding, widened back to the activation dtype."""
    return mx.from_fp8(mx.to_fp8(x), dtype=x.dtype)


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __call__(self, up, gate):
        dtype = up.dtype
        gate = nn.silu(gate.astype(mx.float32))
        up = up.astype(mx.float32)
        if self.limit:
            gate = mx.minimum(gate, self.limit)
            up = mx.clip(up, -self.limit, self.limit)
        return (gate * up).astype(dtype)


class DenseMLP(MLP):
    def __call__(self, x):
        return self.down_proj(
            ClampedSwiGLU(self.limit)(self.up_proj(x), self.gate_proj(x))
        )


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.moe_num_experts, bias=False)
        self.router_bias = mx.zeros(config.moe_num_experts)
        self.top_k = config.moe_top_k
        self.scale = config.moe_router_scaling_factor
        self.normalize = config.norm_expert_weight

    def __call__(self, x):
        logits = self.gate(x.astype(mx.float32))
        scores = mx.sigmoid(logits)
        corrected = scores + self.router_bias
        indices = mx.argsort(-corrected, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(corrected, indices, axis=-1)
        weights = weights - self.router_bias[indices]
        if self.normalize:
            weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
        return indices, weights * self.scale


class MoE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        limit = (config.swiglu_limits or [7.0] * config.num_hidden_layers)[layer_idx]
        shared = (config.swiglu_limits_shared or [7.0] * config.num_hidden_layers)[
            layer_idx
        ]
        self.gate = MoEGate(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.moe_num_experts,
            activation=ClampedSwiGLU(limit),
        )
        self.share_expert = DenseMLP(config, config.share_expert_dim, shared)

    def __call__(self, x):
        indices, weights = self.gate(x)
        routed = self.switch_mlp(x, indices).astype(mx.float32)
        # Match the reference's top-k slot accumulation and BF16 boundary.
        output = mx.zeros(x.shape, dtype=mx.float32)
        for i in range(indices.shape[-1]):
            output = output + routed[..., i, :] * weights[..., i, None]
        return output.astype(x.dtype).astype(mx.float32) + self.share_expert(x).astype(
            mx.float32
        )


class StepRoPE(nn.Module):
    """Partial NEOX rotation with BF16 cosine/sine tables, as in StepFun."""

    def __init__(self, dims, theta, scaling=None):
        super().__init__()
        self.dims = dims
        inv = 1.0 / theta ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        if scaling:
            kind = scaling.get("rope_type", scaling.get("type"))
            if kind != "llama3":
                raise ValueError(f"Unsupported Step 5 RoPE scaling: {kind}")
            factor = scaling["factor"]
            low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
            original = scaling["original_max_position_embeddings"]
            wavelength = 2 * mx.pi / inv
            scaled = mx.where(wavelength > original / low, inv / factor, inv)
            smooth = (original / wavelength - low) / (high - low)
            inv = mx.where(
                (wavelength >= original / high) & (wavelength <= original / low),
                (1 - smooth) * scaled / factor + smooth * scaled,
                scaled,
            )
        self._inv = inv
        mx.eval(self._inv)

    def __call__(self, x, offset=0):
        angle = (mx.arange(x.shape[-2], dtype=mx.float32) + offset)[:, None] * self._inv
        c = mx.cos(angle).astype(mx.bfloat16).astype(mx.float32)
        s = mx.sin(angle).astype(mx.bfloat16).astype(mx.float32)
        a = x[..., : self.dims // 2].astype(mx.float32)
        b = x[..., self.dims // 2 : self.dims].astype(mx.float32)
        return mx.concatenate(
            [
                (a * c - b * s).astype(x.dtype),
                (a * s + b * c).astype(x.dtype),
                x[..., self.dims :],
            ],
            axis=-1,
        )


class Attention(StepAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        theta = (
            config.rope_theta[layer_idx]
            if isinstance(config.rope_theta, list)
            else config.rope_theta
        )
        scaling = (
            None if self.is_sliding and config.yarn_only_types else config.rope_scaling
        )
        partial = (
            config.partial_rotary_factors[layer_idx]
            if config.partial_rotary_factors
            else 1.0
        )
        self.rope = StepRoPE(int(self.head_dim * partial), theta, scaling)
        cfg = config.sparse_config or {}
        layer_type = "sliding_attention" if self.is_sliding else "full_attention"
        self.is_sparse = cfg.get("enabled", False) and layer_type in cfg.get(
            "apply_to_layer_types", ["full_attention"]
        )
        if not self.is_sparse:
            return
        self.proxy_dim = cfg.get("proxy_dim", 256)
        self.index_heads = cfg.get("sparse_indexer_num_heads", 16)
        self.region_size = cfg.get("region_block_size", 8)
        self.topk = cfg.get("topk", 512)
        if (
            cfg.get("num_provider_groups", self.n_kv_heads) != self.n_kv_heads
            or self.index_heads % self.n_kv_heads
        ):
            raise ValueError("Step 5 indexer provider groups must match the KV heads")
        self.sparse_indexer_q = nn.Linear(
            config.hidden_size, self.index_heads * self.proxy_dim, bias=False
        )
        self.sparse_indexer_k = nn.Linear(
            config.hidden_size, self.proxy_dim, bias=False
        )
        self.sparse_indexer_z = nn.Linear(
            config.hidden_size, self.proxy_dim, bias=False
        )
        self.sparse_indexer_w = nn.Linear(
            config.hidden_size, self.index_heads, bias=False
        )
        self.sparse_indexer_q_norm = StepRMSNorm(self.proxy_dim, config.rms_norm_eps)
        self.sparse_indexer_k_norm = nn.LayerNorm(
            self.proxy_dim, eps=config.rms_norm_eps
        )
        self.index_rope = StepRoPE(
            cfg.get("sparse_indexer_rope_dim", 32), theta, scaling
        )
        # Retained checkpoint metadata; the published inference path does not use it.
        self.ssmax_s = mx.zeros(self.n_heads)

    def _sparse_mask(self, x, cache, offset):
        b, length, _ = x.shape
        q = self.sparse_indexer_q(x).reshape(
            b, length, self.index_heads, self.proxy_dim
        )
        # The deployed indexer rounds norm parameters to the activation dtype
        # before its FP32 normalization. Undo the checkpoint's +1 conversion
        # around that rounding boundary for the zero-centered Q norm.
        q_weight = (self.sparse_indexer_q_norm.weight - 1).astype(x.dtype)
        q = (
            mx.fast.rms_norm(
                q.astype(mx.float32),
                q_weight.astype(mx.float32) + 1,
                self.sparse_indexer_q_norm.eps,
            )
            .astype(x.dtype)
            .transpose(0, 2, 1, 3)
        )
        q = round_e4m3(self.index_rope(q, offset))
        k = self.sparse_indexer_k(x).astype(mx.float32)
        mean = k.mean(axis=-1, keepdims=True)
        variance = (k * k).mean(axis=-1, keepdims=True) - mean * mean
        norm = self.sparse_indexer_k_norm
        k = (k - mean) * mx.rsqrt(variance + norm.eps)
        k = k * norm.weight.astype(x.dtype).astype(mx.float32) + norm.bias.astype(
            x.dtype
        ).astype(mx.float32)
        k = k.astype(x.dtype)[:, None]
        k = self.index_rope(k, offset)
        z = self.sparse_indexer_z(x)[:, None]
        if cache is not None:
            k, z = cache.update_and_fetch(k, z)
        total = k.shape[-2]
        regions = total // self.region_size
        positions = mx.arange(offset, offset + length)
        key_positions = mx.arange(total)
        causal = key_positions[None, :] <= positions[:, None]
        if regions <= self.topk:
            return causal
        k = k[..., : regions * self.region_size, :].reshape(
            b, regions, self.region_size, self.proxy_dim
        )
        z = z[..., : regions * self.region_size, :].reshape(k.shape).astype(mx.float32)
        exp = mx.exp(z - mx.max(z, axis=(-2, -1), keepdims=True))
        summaries = (exp * k.astype(mx.float32)).sum(-2) / mx.maximum(
            exp.sum(-2), 1e-20
        )
        summaries = round_e4m3(summaries.astype(mx.bfloat16)).astype(mx.float32)
        per_group = self.index_heads // self.n_kv_heads
        score = q.astype(mx.float32) @ summaries[:, None].swapaxes(-1, -2)
        weight = (x @ self.sparse_indexer_w.weight.astype(x.dtype).T).astype(mx.float32)
        score = mx.maximum(score, 0) * weight.transpose(0, 2, 1)[..., None]
        score = (
            score.reshape(b, self.n_kv_heads, per_group, length, regions).sum(2)
            * per_group**-0.5
        )
        history = mx.arange(regions)[None, :] < (positions // self.region_size)[:, None]
        score = mx.where(history, score, -float("inf"))
        # Stable sorting breaks equal-score ties toward the earlier region.
        chosen = mx.argsort(-score, axis=-1)[..., : self.topk]
        selected = mx.zeros(score.shape, dtype=mx.bool_)
        selected = mx.put_along_axis(selected, chosen, mx.array(True), axis=-1)
        if total % self.region_size:
            selected = mx.pad(selected, [(0, 0), (0, 0), (0, 0), (0, 1)])
        selected = selected[..., key_positions // self.region_size]
        current = (
            key_positions[None, :] // self.region_size
            == positions[:, None] // self.region_size
        )
        selected = (selected | current) & causal
        return mx.repeat(selected, self.n_heads // self.n_kv_heads, axis=1)

    def __call__(self, x, mask=None, cache=None):
        if not self.is_sparse:
            return super().__call__(x, mask=mask, cache=cache)
        kv, index = (cache[0], cache[1]) if cache is not None else (None, None)
        offset = kv.offset if kv is not None else 0
        sparse_mask = self._sparse_mask(x, index, offset)
        return super().__call__(x, mask=sparse_mask, cache=kv)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.is_sliding = self.self_attn.is_sliding
        layers = config.moe_layers_enum
        layers = (
            [int(i) for i in layers.split(",") if i]
            if isinstance(layers, str)
            else layers
        )
        limit = (config.swiglu_limits or [7.0] * config.num_hidden_layers)[layer_idx]
        self.mlp = (
            MoE(config, layer_idx)
            if layer_idx in layers
            else DenseMLP(config, config.intermediate_size, limit)
        )
        self.input_layernorm = StepRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = StepRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        dtype = self.self_attn.q_proj.weight.dtype
        if dtype == mx.uint32:
            dtype = mx.bfloat16
        hidden = self.input_layernorm(x).astype(dtype)
        attn = self.self_attn(hidden, mask=mask, cache=cache)
        h = x + attn.astype(x.dtype)
        hidden = self.post_attention_layernorm(h).astype(dtype)
        return h + self.mlp(hidden).astype(h.dtype)


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = StepRMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, input_ids, inputs_embeds=None, cache=None):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        dtype = h.dtype
        if self.args.fp32_residual_connection:
            h = h.astype(mx.float32)
        for layer, c in zip(self.layers, cache or [None] * len(self.layers)):
            kv = c[0] if isinstance(c, CacheList) else c
            mask = create_attention_mask(
                h,
                kv,
                window_size=self.args.sliding_window if layer.is_sliding else None,
            )
            h = layer(h, mask=mask, cache=c)
        return self.norm(h.astype(dtype))


class LanguageModel(StepLanguageModel):
    def __init__(self, args, config=None):
        nn.Module.__init__(self)
        self.args = args
        self.model = TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def make_cache(self):
        return [
            (
                CacheList(KVCache(), KVCache())
                if layer.self_attn.is_sparse
                else (
                    RotatingKVCache(max_size=self.args.sliding_window)
                    if layer.is_sliding
                    else KVCache()
                )
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The router and sparse selection weights are evaluated with
            # specific FP32/BF16 boundaries in the reference implementation.
            return not (".gate.gate" in path or ".sparse_indexer_w" in path)

        return predicate

    def sanitize(self, weights):
        out = {}
        for name, value in weights.items():
            if (
                ".layers." in name
                and int(name.split(".layers.")[1].split(".")[0])
                >= self.args.num_hidden_layers
            ):
                continue
            out[_remap_language_key(name)] = value
        return out
