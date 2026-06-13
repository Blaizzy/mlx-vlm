"""MiniMax-M3 text tower for mlx-vlm.

Phase 1: full attention on every layer. That is numerically identical to M3's MiniMax Sparse
Attention at normal context (the Lightning Indexer only starts dropping key-blocks past ~2K
tokens), so this is correct for typical prompts. Phase 2 will add the block-sparse indexer for
long-context efficiency. Ported from mlx_lm/models/minimax.py + the transformers M3 reference.
"""
import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.switch_layers import SwitchGLU

from .config import TextConfig


def sanitize(weights):
    """Map M3 checkpoint names -> this module's names:
      * drop the Lightning-Indexer weights (Phase 1 uses full attention)
      * rename `block_sparse_moe` -> `mlp`
      * stack per-expert w1/w2/w3 -> switch_mlp.{gate,down,up}_proj
    """
    weights = {re.sub(r"self_attn\.index_(q|k)_(proj|norm)", r"self_attn.indexer.\1_\2", k): v
               for k, v in weights.items()}
    weights = {k.replace("block_sparse_moe", "mlp"): v for k, v in weights.items()}
    pat = re.compile(r"(.*\.layers\.\d+)\.mlp\.experts\.(\d+)\.(w1|w2|w3)\.weight")
    nm = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    out, buckets = {}, {}
    for k, v in weights.items():
        m = pat.match(k)
        if m:
            buckets.setdefault((m.group(1), nm[m.group(3)]), {})[int(m.group(2))] = v
        else:
            out[k] = v
    for (prefix, new), exps in buckets.items():
        out[f"{prefix}.mlp.switch_mlp.{new}.weight"] = mx.stack([exps[e] for e in range(len(exps))])
    return out


class SwigluOAI(nn.Module):
    """GPT-OSS / M3 clamped SwiGLU activation, given separate up and gate projections."""

    def __init__(self, alpha: float, limit: float):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x_up, x_gate):
        gate = mx.minimum(x_gate, self.limit)
        up = mx.clip(x_up, -self.limit, self.limit)
        return (up + 1.0) * (gate * mx.sigmoid(self.alpha * gate))


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: scale by (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class M3MLP(nn.Module):
    """swigluoai with SEPARATE gate/up/down projections (matches checkpoint mlp.* / shared_experts.*)."""

    def __init__(self, config: TextConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

    def __call__(self, x):
        gate = mx.minimum(self.gate_proj(x), self.limit)
        up = mx.clip(self.up_proj(x), -self.limit, self.limit)
        glu = gate * mx.sigmoid(self.alpha * gate)
        return self.down_proj((up + 1.0) * glu)


class M3MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.routed_scaling = config.routed_scaling_factor
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((config.num_local_experts,))
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.intermediate_size, config.num_local_experts,
            activation=SwigluOAI(config.swiglu_alpha, config.swiglu_limit),
        )
        self.shared_experts = M3MLP(config, config.shared_intermediate_size)

    def __call__(self, x):
        shared = self.shared_experts(x)
        weights = mx.sigmoid(self.gate(x).astype(mx.float32))
        scores = weights + self.e_score_correction_bias
        idx = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        tw = mx.take_along_axis(weights, idx, axis=-1)
        tw = (tw / (tw.sum(axis=-1, keepdims=True) + 1e-20)).astype(x.dtype)
        y = self.switch_mlp(x, idx)                  # [B, L, top_k, D]
        y = (y * tw[..., None]).sum(axis=-2)         # [B, L, D]
        return y * self.routed_scaling + shared


class M3Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int = 0):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.rotary_dim = config.rotary_dim
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.rotary_dim, traditional=False, base=config.rope_theta)
        self.layer_idx = layer_idx
        if config.is_sparse_layer(layer_idx):
            from .msa import M3Indexer
            self.indexer = M3Indexer(config)
            self.block = config.index_block_size
        else:
            self.indexer = None

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)
        if self.indexer is not None and cache is None and L > self.block:
            from .msa import block_sparse_attention
            sel = self.indexer.select_per_qblock(x)
            nqb = -(-L // self.block); P = nqb * self.block - L
            if P:
                q = mx.concatenate([q, mx.zeros((B, self.n_heads, P, self.head_dim), dtype=q.dtype)], axis=2)
                k = mx.concatenate([k, mx.zeros((B, self.n_kv, P, self.head_dim), dtype=k.dtype)], axis=2)
                v = mx.concatenate([v, mx.zeros((B, self.n_kv, P, self.head_dim), dtype=v.dtype)], axis=2)
            o = block_sparse_attention(q, k, v, sel, self.block, self.scale, valid_len=L)
            o = o[:, :, :L, :].transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(o)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class M3DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = M3Attention(config, layer_idx)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if layer_idx < config.first_k_dense:
            self.mlp = M3MLP(config, config.dense_intermediate_size)
        else:
            self.mlp = M3MoE(config)

    def __call__(self, x, mask=None, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMaxM3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [M3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, mask=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


from ..base import LanguageModelOutput


class LanguageModel(nn.Module):
    """Top text wrapper: embedding+layers+norm (MiniMaxM3Model) plus the LM head.

    Accepts pre-merged `inputs_embeds` (image tokens already scattered in) from the VL Model,
    or raw `input_ids` for text-only use (the strategist path).
    """

    def __init__(self, config: TextConfig, model_config=None):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = MiniMaxM3Model(config)
        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids=None, inputs_embeds=None, mask=None, cache=None, **kwargs):
        out = self.model(inputs=input_ids, cache=cache, inputs_embeds=inputs_embeds, mask=mask)
        if getattr(self.config, "tie_word_embeddings", False):
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        return sanitize(weights)
