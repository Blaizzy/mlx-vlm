"""Youtu-Parsing-Omni language model: 40-layer dense MLA decoder with an untied lm_head.

Ported from tencent/Youtu-Parsing-Omni (https://huggingface.co/tencent/Youtu-Parsing-Omni).
Copyright (C) 2026 Tencent. The original code is distributed under the Youtu-Parsing
license (https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/main/LICENSE);
modified / re-implemented here for MLX.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask

# The MLA attention and the dense SwiGLU MLP are byte-identical to youtu_vl's, so they
# are reused (same parameter names / numerics). The decoder layer and model are kept
# here because youtu_vl's layer carries an MoE branch driven by config fields that
# this (dense) config does not have.
from ..youtu_vl.language import YoutuAttention as YoutuVITAAttention
from ..youtu_vl.language import YoutuMLP as YoutuVITAMLP
from .config import ModelConfig, TextConfig


class YoutuVITADecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = YoutuVITAAttention(config)

        self.mlp = YoutuVITAMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class YoutuVITAModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            YoutuVITADecoderLayer(config=config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig, model_config: ModelConfig = None):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.model_type = config.model_type
        self.model = YoutuVITAModel(config)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds, mask=mask)
        k = kwargs.get("logits_to_keep")
        if k is not None:
            out = out[:, -k:, :]
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.qk_nope_head_dim + self.config.qk_rope_head_dim

    @property
    def n_kv_heads(self):
        # MLA caches a single shared latent head (text_config's 8 is misleading)
        return 1
