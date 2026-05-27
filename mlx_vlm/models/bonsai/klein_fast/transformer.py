from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlx_vlm.models.bonsai.constants import ModelConfig
from mlx_vlm.models.bonsai.klein_fast.blocks import DEFAULT_QUANT_GROUP_SIZE
from mlx_vlm.models.bonsai.klein_fast.megakernel import (
    Flux2KleinMegakernel,
    Flux2KleinMegakernelSpec,
    MegakernelWeights,
    PrecisionName,
    eval_megakernel_constants,
)
from mlx_vlm.models.bonsai.pos_embed import Flux2PosEmbed
from mlx_vlm.models.bonsai.time_embed import Flux2TimestepGuidanceEmbeddings


class Flux2KleinFastTransformer(nn.Module):
    def __init__(
        self,
        *,
        weights: MegakernelWeights,
        precision: PrecisionName,
        group_size: int = DEFAULT_QUANT_GROUP_SIZE,
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
        layer_norm_eps: float = 1e-6,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            guidance_embeds=guidance_embeds,
        )

        spec = Flux2KleinMegakernelSpec(
            num_double_blocks=num_layers,
            num_single_blocks=num_single_layers,
            dim=self.inner_dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            layer_norm_eps=layer_norm_eps,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            axes_dims_rope=axes_dims_rope,
            in_channels=in_channels,
            context_dim=joint_attention_dim,
        )
        self.megakernel = Flux2KleinMegakernel(
            spec=spec,
            weights=weights,
            precision=precision,
            group_size=group_size,
        )
        # Materialize all quantized/dense weight arrays so they live as on-device
        # constants rather than lazy subgraphs rebuilt each call.
        eval_megakernel_constants(self.megakernel)

        # Compile the per-step forward once; shape-identical calls hit the cache.
        def _forward_with_modulations(
            hidden_states: mx.array,
            encoder_hidden_states: mx.array,
            temb: mx.array,
            rotary_cos: mx.array,
            rotary_sin: mx.array,
        ) -> mx.array:
            precomputed = self.megakernel.prepare_all_modulations(temb)
            return self.megakernel.forward_from_modulations(
                hidden_states,
                encoder_hidden_states,
                precomputed,
                rotary_cos,
                rotary_sin,
            )

        self._compiled_forward = mx.compile(_forward_with_modulations)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array | float | int,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | float | int | None = None,
    ) -> mx.array:
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

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_cos = mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0)
        concat_sin = mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0)

        return self._compiled_forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            concat_cos,
            concat_sin,
        )
