from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import create_attention_mask
from ..cache import KVCache
from ..deepseek_v3.language import DeepseekV3DecoderLayer
from ..deepseek_v3.language import LanguageModel as DeepseekV3LanguageModel
from ..pipeline import PipelineMixin


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hc = config.hc_mult
        self.hc_fn = mx.zeros(((2 + hc) * hc, hc * config.hidden_size))
        self.hc_base = mx.zeros(((2 + hc) * hc,))
        self.hc_scale = mx.ones((3,))

    def __call__(self, x):
        config = self.config
        hc = config.hc_mult
        flat = x.reshape(*x.shape[:2], -1).astype(mx.float32)
        flat = flat * mx.rsqrt(
            mx.mean(flat * flat, axis=-1, keepdims=True) + config.rms_norm_eps
        )
        mix = (flat.astype(x.dtype) @ self.hc_fn.astype(x.dtype).T).astype(mx.float32)
        pre, post, comb = mx.split(mix, [hc, 2 * hc], axis=-1)
        pre_b, post_b, comb_b = mx.split(self.hc_base, [hc, 2 * hc])
        pre = mx.sigmoid(pre * self.hc_scale[0] + pre_b)
        post = 2 * mx.sigmoid(post * self.hc_scale[1] + post_b)
        comb = comb.reshape(*comb.shape[:-1], hc, hc) * self.hc_scale[
            2
        ] + comb_b.reshape(hc, hc)
        comb = mx.clip(comb, config.mhc_h_res_clamp_min, config.mhc_h_res_clamp_max)
        comb = mx.exp(comb - comb.max(axis=-1, keepdims=True))
        for _ in range(config.hc_sinkhorn_iters):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + config.hc_eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + config.hc_eps)
        collapsed = (pre[..., None].astype(x.dtype) * x).sum(axis=2)
        return post.astype(x.dtype), comb.astype(x.dtype), collapsed


class DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, x, mask=None, cache=None):
        post, comb, collapsed = self.attn_hc(x)
        out = self.self_attn(self.input_layernorm(collapsed), mask, cache)
        x = post[..., None] * out[:, :, None] + comb @ x
        post, comb, collapsed = self.ffn_hc(x)
        out = self.mlp(self.post_attention_layernorm(collapsed))
        return post[..., None] * out[:, :, None] + comb @ x


class Xing4_0Model(PipelineMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: Optional[Any] = None) -> mx.array:
        h = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0], return_array=True)
        h = mx.repeat(h[:, :, None], self.config.hc_mult, axis=2)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        return self.norm(h.mean(axis=2))


class LanguageModel(DeepseekV3LanguageModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = Xing4_0Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def sanitize(self, weights):
        mtp = f"model.layers.{self.args.num_hidden_layers}."
        weights = {k: v for k, v in weights.items() if not k.startswith(mtp)}
        return super().sanitize(weights)

    def make_cache(self):
        return [KVCache() for _ in self.model.layers]
