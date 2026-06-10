import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.transformer.attention import Flux2Attention
from mlx_vlm.models.flux2.transformer.feed_forward import Flux2FeedForward
from mlx_vlm.models.flux2.transformer.kv_cache import Flux2KVLayerCache


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.norm1_context = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux2Attention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            added_kv_proj_dim=dim,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff = Flux2FeedForward(dim=dim, mult=mlp_ratio)
        self.norm2_context = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff_context = Flux2FeedForward(dim=dim, mult=mlp_ratio)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb_mod_params_img,
        temb_mod_params_txt,
        image_rotary_emb,
        *,
        kv_cache: Flux2KVLayerCache | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
    ):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
            temb_mod_params_img
        )
        (c_shift_msa, c_scale_msa, c_gate_msa), (
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        attn_output, encoder_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_ref_tokens=num_ref_tokens,
        )

        hidden_states = hidden_states + gate_msa * attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * encoder_attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (1 + scale_mlp) * norm_hidden_states + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_mlp
        ) * norm_encoder_hidden_states + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(
            norm_encoder_hidden_states
        )

        return encoder_hidden_states, hidden_states
