import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.transformer.ada_layer_norm_continuous import (
    AdaLayerNormContinuous,
)
from mlx_vlm.models.flux2.transformer.kv_cache import Flux2KVCache
from mlx_vlm.models.flux2.transformer.modulation import Flux2Modulation
from mlx_vlm.models.flux2.transformer.pos_embed import Flux2PosEmbed
from mlx_vlm.models.flux2.transformer.single_transformer_block import (
    Flux2SingleTransformerBlock,
)
from mlx_vlm.models.flux2.transformer.timestep_guidance_embeddings import (
    Flux2TimestepGuidanceEmbeddings,
)
from mlx_vlm.models.flux2.transformer.transformer_block import Flux2TransformerBlock


class Flux2Transformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 5,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 7680,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        guidance_embeds: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            guidance_embeds=guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim, mod_param_sets=2
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim, mod_param_sets=2
        )
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim, mod_param_sets=1
        )

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(
            joint_attention_dim, self.inner_dim, bias=False
        )
        self.transformer_blocks = [
            Flux2TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ]
        self.single_transformer_blocks = [
            Flux2SingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_single_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=False
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array | float | int,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | float | int | None = None,
        kv_cache: Flux2KVCache | None = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ) -> mx.array | tuple[mx.array, Flux2KVCache]:
        num_txt_tokens = encoder_hidden_states.shape[1]
        num_img_tokens = hidden_states.shape[1]
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep, dtype=hidden_states.dtype)
        if timestep.ndim == 0:
            timestep = mx.full(
                (hidden_states.shape[0],), timestep, dtype=hidden_states.dtype
            )
        timestep = timestep.astype(hidden_states.dtype)
        timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(
            hidden_states.dtype
        )
        timestep = timestep * timestep_scale
        if guidance is not None:
            if not isinstance(guidance, mx.array):
                guidance = mx.array(guidance, dtype=hidden_states.dtype)
            if guidance.ndim == 0:
                guidance = mx.full(
                    (hidden_states.shape[0],), guidance, dtype=hidden_states.dtype
                )
            guidance = guidance.astype(hidden_states.dtype)
            guidance_scale = mx.where(mx.max(guidance) <= 1.0, 1000.0, 1.0).astype(
                hidden_states.dtype
            )
            guidance = guidance * guidance_scale
        temb = self.time_guidance_embed(timestep, guidance)
        temb = temb.astype(ModelConfig.precision)
        ref_temb = None
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            if kv_cache is None:
                kv_cache = Flux2KVCache(
                    num_double_layers=len(self.transformer_blocks),
                    num_single_layers=len(self.single_transformer_blocks),
                )
            kv_cache.num_ref_tokens = num_ref_tokens
            ref_timestep = mx.full(
                timestep.shape,
                ref_fixed_timestep * 1000.0,
                dtype=timestep.dtype,
            )
            ref_temb = self.time_guidance_embed(ref_timestep, guidance).astype(
                ModelConfig.precision
            )

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
            mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
        )

        temb_mod_params_img = self.double_stream_modulation_img(temb)
        temb_mod_params_txt = self.double_stream_modulation_txt(temb)
        if ref_temb is not None:
            ref_mod_params_img = self.double_stream_modulation_img(ref_temb)
            temb_mod_params_img = _blend_double_mod_params(
                temb_mod_params_img,
                ref_mod_params_img,
                num_ref_tokens=num_ref_tokens,
                seq_len=num_img_tokens,
            )

        for index, block in enumerate(self.transformer_blocks):
            layer_cache = (
                kv_cache.get_double(index)
                if kv_cache_mode is not None and kv_cache is not None
                else None
            )
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=concat_rotary_emb,
                kv_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                num_ref_tokens=num_ref_tokens,
            )

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        temb_mod_params_single = self.single_stream_modulation(temb)[0]
        if ref_temb is not None:
            ref_mod_params_single = self.single_stream_modulation(ref_temb)[0]
            temb_mod_params_single = _blend_single_mod_params(
                temb_mod_params_single,
                ref_mod_params_single,
                num_txt_tokens=num_txt_tokens,
                num_ref_tokens=num_ref_tokens,
                seq_len=hidden_states.shape[1],
            )
        for index, block in enumerate(self.single_transformer_blocks):
            layer_cache = (
                kv_cache.get_single(index)
                if kv_cache_mode is not None and kv_cache is not None
                else None
            )
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod_params=temb_mod_params_single,
                image_rotary_emb=concat_rotary_emb,
                kv_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                num_txt_tokens=num_txt_tokens,
                num_ref_tokens=num_ref_tokens,
            )

        start = num_txt_tokens
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            start += num_ref_tokens
        hidden_states = hidden_states[:, start:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        if kv_cache_mode == "extract":
            if kv_cache is None:
                raise RuntimeError("Flux2 KV cache was not initialized")
            return hidden_states, kv_cache
        return hidden_states


def _expand_mod_param(param: mx.array, seq_len: int) -> mx.array:
    if param.ndim == 2:
        param = mx.expand_dims(param, axis=1)
    if param.shape[1] == seq_len:
        return param
    return mx.broadcast_to(param, (param.shape[0], seq_len, param.shape[2]))


def _blend_mod_param_set(
    img_params: tuple[mx.array, mx.array, mx.array],
    ref_params: tuple[mx.array, mx.array, mx.array],
    *,
    num_ref_tokens: int,
    seq_len: int,
) -> tuple[mx.array, mx.array, mx.array]:
    blended = []
    for img_param, ref_param in zip(img_params, ref_params):
        img_expanded = _expand_mod_param(img_param, seq_len)
        ref_expanded = _expand_mod_param(ref_param, num_ref_tokens)
        blended.append(
            mx.concatenate([ref_expanded, img_expanded[:, num_ref_tokens:, :]], axis=1)
        )
    return tuple(blended)


def _blend_double_mod_params(
    img_mod_params,
    ref_mod_params,
    *,
    num_ref_tokens: int,
    seq_len: int,
):
    return tuple(
        _blend_mod_param_set(
            img_set,
            ref_set,
            num_ref_tokens=num_ref_tokens,
            seq_len=seq_len,
        )
        for img_set, ref_set in zip(img_mod_params, ref_mod_params)
    )


def _blend_single_mod_params(
    img_params: tuple[mx.array, mx.array, mx.array],
    ref_params: tuple[mx.array, mx.array, mx.array],
    *,
    num_txt_tokens: int,
    num_ref_tokens: int,
    seq_len: int,
) -> tuple[mx.array, mx.array, mx.array]:
    blended = []
    for img_param, ref_param in zip(img_params, ref_params):
        img_expanded = _expand_mod_param(img_param, seq_len)
        ref_expanded = _expand_mod_param(ref_param, num_ref_tokens)
        blended.append(
            mx.concatenate(
                [
                    img_expanded[:, :num_txt_tokens, :],
                    ref_expanded,
                    img_expanded[:, num_txt_tokens + num_ref_tokens :, :],
                ],
                axis=1,
            )
        )
    return tuple(blended)
