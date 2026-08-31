from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import CacheList, KVCache
from ..deepseek_v32.language import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from ..deepseek_v32.language import Model as DSV32Model
from .config import ModelConfig
from .speculative_verifier import GlmMoeDsaExactSpeculativeVerifier, verify_logits

_SPECULATIVE_VERIFIER = GlmMoeDsaExactSpeculativeVerifier()


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices

        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(x) if inputs_embeds is None else inputs_embeds

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = GlmMoeDsaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        capture_layer_ids = kwargs.get("capture_layer_ids")
        speculative_verify = bool(kwargs.get("speculative_verify", False))
        return_hidden = kwargs.get("return_hidden", False)
        return_shared_kv = kwargs.get("return_shared_kv", False)
        skip_logits = kwargs.get("skip_logits", False)

        if speculative_verify:
            return _SPECULATIVE_VERIFIER(
                self,
                inputs,
                cache=cache,
                inputs_embeds=inputs_embeds,
                capture_layer_ids=capture_layer_ids,
                return_hidden=return_hidden,
                return_shared_kv=return_shared_kv,
                skip_logits=skip_logits,
            )

        hidden = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )
        num_logits_to_keep = kwargs.get("num_logits_to_keep", 0)
        if num_logits_to_keep:
            hidden = hidden[:, -num_logits_to_keep:, :]
        return LanguageModelOutput(
            logits=None if skip_logits else self.lm_head(hidden),
            hidden_states=[hidden] if return_hidden else None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_draft_hidden(self, hidden: mx.array) -> mx.array:
        return hidden

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return verify_logits(self, hidden)

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> mx.array:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, out.gdn_states

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        out = self(
            inputs,
            cache=cache,
            capture_layer_ids=[],
            speculative_verify=True,
            return_hidden=True,
            return_shared_kv=True,
        )
        return (
            out.hidden_states[-1],
            out.shared_kv_states,
            out.gdn_states,
            sampler(out.logits),
        )

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Optional[List],
        accepted,
        block_size: int,
    ) -> int:
        del gdn_states
        if isinstance(accepted, int):
            accepted_values = [accepted]
        elif isinstance(accepted, mx.array):
            accepted_values = [int(value) for value in accepted.reshape(-1).tolist()]
        else:
            accepted_values = [int(value) for value in accepted]
        if len(set(accepted_values)) != 1:
            raise ValueError("glm_moe_dsa requires uniform batch acceptance.")

        max_accepted = accepted_values[0]
        trim = int(block_size) - (max_accepted + 1)
        if trim > 0:
            for cache in caches:
                if cache is not None and cache.is_trimmable():
                    cache.trim(trim)
        return max_accepted

    requires_uniform_dflash_acceptance = True

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return DSV32Model.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        return DSV32Model.shard(self, group)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return DSV32Model.cast_predicate(self)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
