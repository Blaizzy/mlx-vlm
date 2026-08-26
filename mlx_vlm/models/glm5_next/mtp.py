from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..deepseek_v32.language import DeepseekV32MoE
from .config import TextConfig
from .language import Glm5NextSparseAttention


class Glm5NextMTP(nn.Module):
    """Multi-token-prediction (nextn) head. Drafts token t+2 from the base model's
    hidden state h(t+1) and the embedding of the accepted token t+1. Structure mirrors
    the GLM-5-Next layer-45 nextn layer: enorm/hnorm -> eh_proj -> DSA+MoE decoder ->
    shared_head norm -> (shared) lm_head applied by the caller."""

    def __init__(self, config: TextConfig):
        super().__init__()
        h = config.hidden_size
        self.enorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * h, h, bias=False)
        self.input_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.self_attn = Glm5NextSparseAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.mlp = DeepseekV32MoE(config)
        self.shared_head_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        next_embed: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = self.eh_proj(
            mx.concatenate([self.enorm(next_embed), self.hnorm(hidden)], axis=-1)
        )
        x = x + self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.shared_head_norm(x)


def load_mtp_weights(config: TextConfig, weights: dict, layer_idx: int = 45) -> dict:
    """Map raw layer-{layer_idx} checkpoint tensors -> Glm5NextMTP module tree.
    Absorbs raw kv_b_proj into embed_q/unembed_out (kept in the stored dtype; no
    re-quantization). Experts are already stacked into switch_mlp in the checkpoint."""
    pfx = f"language_model.model.layers.{layer_idx}."
    out = {}
    rename = {
        "shared_head.norm.": "shared_head_norm.",
    }
    for k, v in weights.items():
        if not k.startswith(pfx):
            continue
        rest = k[len(pfx) :]
        for a, b in rename.items():
            if rest.startswith(a):
                rest = b + rest[len(a) :]
        out[rest] = v

    kb = "self_attn.kv_b_proj.weight"
    if kb in out:
        v = out.pop(kb)
        nope, vhd = config.qk_nope_head_dim, config.v_head_dim
        nheads = config.num_attention_heads
        v = v.reshape(nheads, nope + vhd, -1)
        out["self_attn.embed_q.weight"] = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
        out["self_attn.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])
    return out
