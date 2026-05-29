import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.transformer.ada_layer_norm_continuous import (
    AdaLayerNormContinuous,
)
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

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=concat_rotary_emb,
            )

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        temb_mod_params_single = self.single_stream_modulation(temb)[0]
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod_params=temb_mod_params_single,
                image_rotary_emb=concat_rotary_emb,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states
