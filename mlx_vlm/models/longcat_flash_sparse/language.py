import math
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from ..mla import MultiLinear, latent_length, max_absorbed_queries
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig
from .indexer_kernel import indexer_dense_scores, indexer_dense_scores_available


class NgramEmbedding(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.m = args.oe_vocab_size_ratio * args.vocab_size
        self.k = args.oe_split_num
        self.n = args.oe_neighbor_num

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        self.num_embedders = self.k * (self.n - 1)
        emb_dim = args.hidden_size // self.num_embedders
        self.embedders = []
        for i in range(self.num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            self.embedders.append(nn.Embedding(emb_vocab_size, emb_dim))
        # the per-embedder projections fuse into one GEMM over concatenated lookups
        self.post_proj = nn.Linear(
            self.num_embedders * emb_dim, args.hidden_size, bias=False
        )
        self._compute_vocab_mods()

    def _compute_vocab_mods(self):
        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods = vocab_mods

    def _shift_right(self, x: mx.array, n: int) -> mx.array:
        if n <= 0:
            return x
        batch_size, seq_len = x.shape
        if seq_len <= n:
            return mx.zeros_like(x)
        return mx.concatenate(
            [mx.zeros((batch_size, n), dtype=x.dtype), x[..., :-n]], axis=-1
        )

    def _get_ngram_ids(self, input_ids, shifted_ids, vocab_mods, ngram):
        ngram_ids = input_ids
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def __call__(self, input_ids: mx.array, cache: Optional[Any] = None) -> mx.array:
        seq_len = input_ids.shape[-1]
        input_ids = input_ids.astype(mx.int64)
        if cache is not None:
            context = cache[0]
            context = (
                input_ids
                if context is None
                else mx.concatenate([context, input_ids], axis=-1)
            )
            cache[0] = context[..., max(0, context.shape[-1] - self.n + 1) :]
        else:
            context = input_ids

        x = self.word_embeddings(input_ids)
        shifted_ids = {
            i: self._shift_right(context, i - 1) for i in range(2, self.n + 1)
        }
        lookups = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, self._vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                lookups.append(self.embedders[index](new_ids))
        # LongcatCausalLM (Lite-Sparse) keeps the word embedding at full scale and
        # normalizes only the n-gram contribution, unlike LongcatFlashNgram which
        # divides the whole sum by (1 + num_embedders).
        proj = self.post_proj(mx.concatenate(lookups, axis=-1))
        return x + proj / (1 + self.num_embedders)


class Indexer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.index_init_tokens = args.index_init_tokens
        self.index_local_tokens = args.index_local_tokens
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        if args.index_k_norm_type == "rms":
            self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        else:
            self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.indexer_rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        b, s, _ = x.shape
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_norm(self.wk(x)).reshape(b, 1, s, self.head_dim)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))

        seqlen = k.shape[2]
        if seqlen <= self.index_topk:
            return None

        wp = self.weights_proj(x)
        scale = self.n_heads**-0.5 * self.softmax_scale
        if indexer_dense_scores_available(q.dtype, self.n_heads, self.head_dim):
            scores = indexer_dense_scores(q, k[:, 0], wp, scale)[:, None]
        else:
            scores = mx.maximum(q @ k.swapaxes(-1, -2), 0)
            w = (wp * scale).swapaxes(-1, -2)[..., None]
            scores = (scores * w).sum(axis=1, keepdims=True)

        if self.index_init_tokens > 0 or self.index_local_tokens > 0:
            col = mx.arange(seqlen)
            forced = col[None, :] < self.index_init_tokens
            if self.index_local_tokens > 0:
                row_pos = offset + mx.arange(s)
                local = (col[None, :] <= row_pos[:, None]) & (
                    col[None, :] > row_pos[:, None] - self.index_local_tokens
                )
                forced = forced | local
            else:
                forced = mx.broadcast_to(forced, (s, seqlen))
            scores = mx.where(
                forced[None, None], mx.array(float("inf"), scores.dtype), scores
            )

        if mask is not None:
            scores = mx.where(mask, scores, -float("inf"))
        return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
            ..., -self.index_topk :
        ]


class LongcatFlashMLA(nn.Module):
    def __init__(self, args: ModelConfig, is_index_owner: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.v_head_dim = args.v_head_dim
        self.use_lsa = args.attention_method == "LSA"

        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, self.q_lora_rank, bias=args.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_attention_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_attention_heads
        )
        self._absorbed_dims = (
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.v_head_dim,
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.mla_scale_q_lora = None
        self.mla_scale_kv_lora = None
        if args.mla_scale_q_lora:
            self.mla_scale_q_lora = (args.hidden_size / self.q_lora_rank) ** 0.5
        if args.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (args.hidden_size / self.kv_lora_rank) ** 0.5

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = args.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.indexer = Indexer(args) if (self.use_lsa and is_index_owner) else None
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        latent_cache: Optional[Any] = None,
        index_cache: Optional[Any] = None,
        topk_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            qr = None
            q = self.q_proj(x)
        else:
            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_attention_heads, self.qk_head_dim).transpose(
            0, 2, 1, 3
        )
        if self.mla_scale_q_lora is not None:
            q = q * self.mla_scale_q_lora
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        if self.mla_scale_kv_lora is not None:
            kv_latent = kv_latent * self.mla_scale_kv_lora

        offset = latent_cache.offset if latent_cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if latent_cache is not None:
            kv_latent, k_pe = latent_cache.update_and_fetch(kv_latent, k_pe)

        if self.indexer is not None:
            topk_indices = self.indexer(x, qr, mask, cache=index_cache)

        if self.use_lsa and topk_indices is not None:
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
            if latent_cache is not None and index_cache is not None:
                latent_cache.keys = mx.depends(
                    latent_cache.keys, (index_cache.keys, index_cache.values)
                )

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        absorbed = L == 1 or L <= max_absorbed_queries(
            *self._absorbed_dims, latent_length(kv_latent)
        )
        if absorbed:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=latent_cache, scale=self.scale, mask=pe_scores
        )
        if absorbed:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_indices


class LongcatFlashMLP(nn.Module):
    def __init__(self, args: ModelConfig, is_expert: bool = False):
        super().__init__()
        hidden_size = args.expert_ffn_hidden_size if is_expert else args.ffn_hidden_size
        self.gate_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.top_k = args.moe_topk
        self.n_routed_experts = args.n_routed_experts + args.zero_expert_num
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.classifier = nn.Linear(
            args.hidden_size, self.n_routed_experts, bias=args.router_bias
        )
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        dtype = hidden_states.dtype
        scores = mx.softmax(self.classifier(hidden_states), axis=-1)
        corrected_scores = scores + self.e_score_correction_bias
        topk_indices = mx.argpartition(corrected_scores, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)
        if self.norm_topk_prob:
            denominator = mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights.astype(dtype)


class LongcatFlashMoE(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_experts_per_tok = args.moe_topk
        self.n_routed_experts = args.n_routed_experts
        self.zero_expert_num = args.zero_expert_num
        self.zero_expert_type = args.zero_expert_type
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.expert_ffn_hidden_size,
            args.n_routed_experts,
        )
        self.router = LongcatFlashTopkRouter(args)
        self.sharding_group = None

    def __call__(self, hidden_states):
        if self.sharding_group is not None:
            hidden_states = sum_gradients(self.sharding_group)(hidden_states)

        topk_indices, topk_weights = self.router(hidden_states)

        mask = topk_indices >= self.n_routed_experts
        regular_indices = mx.where(mask, 0, topk_indices)
        regular_weights = mx.where(mask, 0.0, topk_weights)

        regular_outputs = self.switch_mlp(hidden_states, regular_indices)
        final_output = mx.sum(regular_outputs * regular_weights[..., None], axis=-2)

        if self.sharding_group is not None:
            final_output = mx.distributed.all_sum(
                final_output, group=self.sharding_group
            )

        if self.zero_expert_num > 0:
            assert self.zero_expert_type == "identity"
            identity_weights_sum = mx.sum(
                mx.where(mask, topk_weights, 0.0), axis=-1, keepdims=True
            )
            final_output = final_output + hidden_states * identity_weights_sum

        return final_output


class LongcatFlashDecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.use_lsa = args.attention_method == "LSA"
        self.mlp = LongcatFlashMoE(args)
        self.self_attn = [
            LongcatFlashMLA(args, is_index_owner=(i == 0)) for i in range(2)
        ]
        self.mlps = [LongcatFlashMLP(args, False) for _ in range(2)]
        self.input_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if cache is None:
            cache = (None, None, None) if self.use_lsa else (None, None)

        hidden_states = x
        shortcut_mlp_output = None
        topk_indices = None

        for i in range(2):
            residual = hidden_states
            h = self.input_layernorm[i](hidden_states)
            if self.use_lsa:
                latent_cache = cache[0] if i == 0 else cache[2]
                index_cache = cache[1] if i == 0 else None
                attn_out, idx = self.self_attn[i](
                    h,
                    mask=mask,
                    latent_cache=latent_cache,
                    index_cache=index_cache,
                    topk_indices=None if i == 0 else topk_indices,
                )
                if i == 0:
                    topk_indices = idx
            else:
                attn_out, _ = self.self_attn[i](h, mask=mask, latent_cache=cache[i])
            hidden_states = residual + attn_out

            residual = hidden_states
            h = self.post_attention_layernorm[i](hidden_states)
            if i == 0:
                shortcut_mlp_output = self.mlp(h)
            hidden_states = residual + self.mlps[i](h)
            if i == 1:
                hidden_states = hidden_states + shortcut_mlp_output

        return hidden_states


class LongcatFlashModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_layers = args.num_layers
        self.use_ngram = args.oe_vocab_size_ratio > 0
        if self.use_ngram:
            self.ngram_embeddings = NgramEmbedding(args)
        else:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LongcatFlashDecoderLayer(args) for _ in range(args.num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.use_ngram:
            if cache is None:
                cache = [None] * (self.num_layers + 1)
            h = self.ngram_embeddings(x, cache=cache[0])
            layer_cache = cache[1:]
        else:
            if cache is None:
                cache = [None] * self.num_layers
            h = self.embed_tokens(x)
            layer_cache = cache

        mask = create_attention_mask(
            h, layer_cache[0][0] if layer_cache[0] else None, return_array=True
        )
        for layer, c in zip(self.layers, layer_cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = LongcatFlashModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        out = self.model(inputs, cache)
        return LanguageModelOutput(logits=self.lm_head(out))

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("classifier"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        if self.args.attention_method == "LSA":
            per_layer = [
                CacheList(KVCache(), KVCache(), KVCache()) for _ in self.model.layers
            ]
        else:
            per_layer = [CacheList(KVCache(), KVCache()) for _ in self.model.layers]
        if self.args.oe_vocab_size_ratio > 0:
            return [ArraysCache(size=1)] + per_layer
        return per_layer

    def sanitize(self, weights):
        for l in range(self.args.num_layers):
            prefix = f"model.layers.{l}"
            for _, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        for l in range(self.args.num_layers):
            for i in range(2):
                prefix = f"model.layers.{l}.self_attn.{i}"
                kv_b_key = f"{prefix}.kv_b_proj.weight"
                if kv_b_key not in weights:
                    continue
                num_heads = self.args.num_attention_heads
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)
                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{prefix}.embed_q.scales"] = wk_s
                    weights[f"{prefix}.embed_q.biases"] = wk_b
                    weights[f"{prefix}.unembed_out.scales"] = wv_s
                    weights[f"{prefix}.unembed_out.biases"] = wv_b
                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

        if self.args.oe_vocab_size_ratio > 0:
            remap = {}
            proj_parts = {}
            for k, v in weights.items():
                if k == "model.embed_tokens.weight":
                    remap["model.ngram_embeddings.word_embeddings.weight"] = v
                elif k.startswith("model.oe_embed_tokens"):
                    i = k[len("model.oe_embed_tokens") :].split(".")[0]
                    remap[f"model.ngram_embeddings.embedders.{i}.weight"] = v
                elif k.startswith("model.oe_embed_proj"):
                    i = int(k[len("model.oe_embed_proj") :].split(".")[0])
                    proj_parts[i] = v
                else:
                    remap[k] = v
            if proj_parts:
                # fuse the per-embedder projections into one GEMM weight
                fused = mx.concatenate(
                    [proj_parts[i] for i in sorted(proj_parts)], axis=1
                )
                remap["model.ngram_embeddings.post_proj.weight"] = fused
            weights = remap

        return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            for attn in layer.self_attn:
                if attn.q_lora_rank is None:
                    attn.q_proj = shard_linear(
                        attn.q_proj, "all-to-sharded", group=group
                    )
                else:
                    attn.q_b_proj = shard_linear(
                        attn.q_b_proj, "all-to-sharded", group=group
                    )
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
                attn.num_attention_heads //= N
                num_heads = attn.num_attention_heads
                sh = rank * num_heads
                eh = sh + num_heads

                def shard_heads(w):
                    return w[sh:eh]

                attn.embed_q.apply(shard_heads)
                attn.unembed_out.apply(shard_heads)

            for mlp in layer.mlps:
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

            layer.mlp.sharding_group = group
            shard_inplace(layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
