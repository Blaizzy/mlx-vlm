import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.transformer.kv_cache import Flux2KVLayerCache
from mlx_vlm.models.flux2.transformer.parallel_self_attention import (
    Flux2ParallelSelfAttention,
)


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux2ParallelSelfAttention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            mlp_ratio=mlp_ratio,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        temb_mod_params,
        image_rotary_emb,
        *,
        kv_cache: Flux2KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
    ):
        mod_shift, mod_scale, mod_gate = temb_mod_params
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(
            norm_hidden_states,
            image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_txt_tokens=num_txt_tokens,
            num_ref_tokens=num_ref_tokens,
        )
        hidden_states = hidden_states + mod_gate * attn_output
        return hidden_states
