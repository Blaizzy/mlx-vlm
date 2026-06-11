import weakref
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchLinear, _gather_sort, _scatter_unsort

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import StaticPrefixKVCache
from ..gemma4.gemma4 import MultimodalEmbedder, masked_scatter
from ..gemma4.language import RMSNormNoScale
from ..gemma4.rope_utils import initialize_rope
from ..gemma4.vision import VisionModel
from .config import ModelConfig, TextConfig


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


def make_compiled_softcap(softcap: float):
    """Fused fp32 upcast + tanh softcap (one pass over the vocab logits)."""

    def _softcap(x):
        return mx.tanh(x.astype(mx.float32) / softcap) * softcap

    return mx.compile(_softcap, shapeless=True)


_EIGHT_BIT_QUANTIZATION = {"group_size": 64, "bits": 8}
_EIGHT_BIT_QUANTIZED_SUFFIXES = (
    "embed_tokens",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def diffusion_gemma_quant_predicate(path, m):
    if not hasattr(m, "to_quantized"):
        return False
    if (
        path.endswith(_EIGHT_BIT_QUANTIZED_SUFFIXES)
        or ".self_attn." in path
        or "router" in path
    ):
        return dict(_EIGHT_BIT_QUANTIZATION)
    return True


class GeGLU(nn.Module):
    def __call__(self, x, gate):
        return geglu(gate, x)


def _cache_offset(cache) -> int:
    if cache is None or getattr(cache, "keys", None) is None:
        return 0
    offset = getattr(cache, "offset", 0)
    if isinstance(offset, mx.array):
        return int(mx.max(offset).item())
    return int(offset)


def _cache_state(cache):
    if cache is None or getattr(cache, "keys", None) is None:
        return None
    if hasattr(cache, "decoder_state"):
        return cache.decoder_state
    if hasattr(cache, "_temporal_order"):
        return cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    return cache.state


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x):
        x = mx.fast.rms_norm(x, None, self.eps)
        x = x * self.scale * self._root_size
        scores = self.proj(x)
        top_k = self.config.top_k_experts
        indices = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = mx.softmax(weights, axis=-1, precise=True)
        weights = weights * self.per_expert_scale[indices]
        return indices, weights


class Experts(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        self.hidden_dims = config.moe_intermediate_size
        self.gate_up_proj = SwitchLinear(
            input_dims=config.hidden_size,
            output_dims=2 * config.moe_intermediate_size,
            num_experts=config.num_experts,
            bias=False,
        )
        self.down_proj = SwitchLinear(
            input_dims=config.moe_intermediate_size,
            output_dims=config.hidden_size,
            num_experts=config.num_experts,
            bias=False,
        )

    def __call__(self, x, top_k_indices, top_k_weights):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = top_k_indices.size >= 64
        indices = top_k_indices
        inv_order = None
        if do_sort:
            x, indices, inv_order = _gather_sort(x, top_k_indices)
        if self.training:
            indices = mx.stop_gradient(indices)

        gate_up = self.gate_up_proj(x, indices, sorted_indices=do_sort)
        gate = gate_up[..., : self.hidden_dims]
        up = gate_up[..., self.hidden_dims :]
        y = self.down_proj(geglu(gate, up), indices, sorted_indices=do_sort)

        if do_sort:
            y = _scatter_unsort(y, inv_order, top_k_indices.shape)

        y = y.squeeze(-2)
        return (y * top_k_weights[..., None]).sum(axis=-2)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = (
            config.num_global_key_value_heads
            if not self.is_sliding and config.num_global_key_value_heads is not None
            else config.num_key_value_heads
        )
        self.scale = 1.0

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = (
            nn.Linear(
                config.hidden_size,
                self.n_kv_heads * self.head_dim,
                bias=config.attention_bias,
            )
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        rope_params = config.rope_parameters.get(self.layer_type, {})
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=False,
            base=rope_params.get("rope_theta", 10000.0),
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        decoder: bool = False,
        offset: Optional[int] = None,
    ):
        B, L, _ = x.shape
        if offset is None:
            offset = _cache_offset(cache)

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = (
            self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self.v_proj is not None
            else keys
        )

        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=offset)
        values = self.v_norm(values).transpose(0, 2, 1, 3)

        if decoder:
            state = _cache_state(cache)
            if state is not None:
                encoder_keys, encoder_values = state
                if self.is_sliding:
                    # The canvas only attends to the last `sliding_window - 1`
                    # encoder positions (the mask already zeroes the rest), so
                    # drop the out-of-window keys/values before SDPA instead of
                    # computing scores for thousands of masked positions. This
                    # keeps the sliding layers O(window) rather than O(context).
                    window = max(self.config.sliding_window - 1, 0)
                    encoder_len = encoder_keys.shape[2]
                    # Only safe when there are no trailing-invalid cache slots
                    # (i.e. the dynamic cache, where offset == encoder_len);
                    # the static-cache window starts before the trailing empties.
                    if window and encoder_len > window and offset >= encoder_len:
                        encoder_keys = encoder_keys[:, :, -window:, :]
                        encoder_values = encoder_values[:, :, -window:, :]
                        if mask is not None and not isinstance(mask, str):
                            mask = mask[..., -(window + L) :]
                keys = mx.concatenate([encoder_keys, keys], axis=2)
                values = mx.concatenate([encoder_values, values], axis=2)
            attn_cache = None
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            attn_cache = cache

        output = scaled_dot_product_attention(
            queries, keys, values, cache=attn_cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.router = Router(config)
        self.experts = Experts(config)
        self.post_feedforward_layernorm_1 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm_2 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm_2 = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        decoder: bool = False,
        offset: Optional[int] = None,
        layer_scalar: Optional[mx.array] = None,
    ):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache, decoder=decoder, offset=offset)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h1 = self.pre_feedforward_layernorm(h)
        h1 = self.mlp(h1)
        h1 = self.post_feedforward_layernorm_1(h1)

        flat = residual.reshape(-1, residual.shape[-1])
        top_k_indices, top_k_weights = self.router(flat)
        h2 = self.pre_feedforward_layernorm_2(flat)
        h2 = self.experts(h2, top_k_indices, top_k_weights)
        h2 = h2.reshape(residual.shape)
        h2 = self.post_feedforward_layernorm_2(h2)

        h = self.post_feedforward_layernorm(h1 + h2)
        h = residual + h
        return h * (self.layer_scalar if layer_scalar is None else layer_scalar)


class EncoderLayerScalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_scalar = mx.ones((1,))


class EncoderLanguageModel(nn.Module):
    def __init__(self, decoder: "DecoderModel"):
        super().__init__()
        # A weakref.ref stays out of the module tree (a proxy would be walked
        # like a real submodule and double-count the decoder weights, e.g. in
        # the wired-limit model size estimate).
        self._decoder_ref = weakref.ref(decoder)
        self.layers = [EncoderLayerScalar() for _ in decoder.layers]

    @property
    def decoder(self):
        return self._decoder_ref()

    @property
    def embed_tokens(self):
        return self.decoder.embed_tokens

    @property
    def norm(self):
        return self.decoder.norm


class SelfConditioning(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.pre_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNormNoScale(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, inputs_embeds, self_conditioning_signal):
        normed = self.pre_norm(self_conditioning_signal)
        signal = self.down_proj(geglu(self.gate_proj(normed), self.up_proj(normed)))
        return self.post_norm(inputs_embeds + signal)


class DecoderModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_conditioning = SelfConditioning(config)

    def _embed_canvas(
        self,
        canvas_ids,
        self_conditioning_logits=None,
        self_conditioning_embeddings=None,
    ):
        inputs_embeds = self.embed_tokens(canvas_ids) * self.embed_scale
        if (
            self_conditioning_logits is not None
            and self_conditioning_embeddings is not None
        ):
            raise ValueError(
                "Only one of self_conditioning_logits or "
                "self_conditioning_embeddings can be set."
            )
        if self_conditioning_embeddings is not None:
            soft_embeddings = self_conditioning_embeddings.astype(inputs_embeds.dtype)
        elif self_conditioning_logits is None:
            soft_embeddings = mx.zeros_like(inputs_embeds)
        else:
            probs = mx.softmax(self_conditioning_logits, axis=-1, precise=True)
            if isinstance(self.embed_tokens, nn.QuantizedEmbedding):
                soft_embeddings = mx.quantized_matmul(
                    probs.astype(inputs_embeds.dtype),
                    self.embed_tokens.weight,
                    self.embed_tokens.scales,
                    self.embed_tokens.biases,
                    transpose=False,
                    group_size=self.embed_tokens.group_size,
                    bits=self.embed_tokens.bits,
                    mode=getattr(self.embed_tokens, "mode", "affine"),
                )
            else:
                soft_embeddings = probs @ self.embed_tokens.weight
            soft_embeddings = (
                soft_embeddings.astype(inputs_embeds.dtype) * self.embed_scale
            )
        return self.self_conditioning(inputs_embeds, soft_embeddings)

    def _make_decoder_masks(self, h, caches, decoder_attention_mask=None):
        if isinstance(decoder_attention_mask, dict):
            return decoder_attention_mask

        B, canvas_length, _ = h.shape
        masks = {}
        for layer_type in set(self.config.layer_types):
            cache = next(
                (
                    c
                    for c, layer in zip(caches or [], self.layers)
                    if layer.layer_type == layer_type
                ),
                None,
            )
            state = _cache_state(cache)
            encoder_len = state[0].shape[2] if state is not None else 0
            valid_encoder_len = min(_cache_offset(cache), encoder_len)
            key_len = encoder_len + canvas_length

            if layer_type == "full_attention":
                if decoder_attention_mask is None:
                    if encoder_len == valid_encoder_len:
                        masks[layer_type] = None
                    else:
                        row = mx.concatenate(
                            [
                                mx.arange(encoder_len) < valid_encoder_len,
                                mx.ones((canvas_length,), dtype=mx.bool_),
                            ],
                            axis=0,
                        )
                        masks[layer_type] = mx.broadcast_to(
                            row[None, None, None, :],
                            (B, 1, canvas_length, key_len),
                        )
                else:
                    full = decoder_attention_mask.astype(mx.bool_)
                    if full.shape[-1] != key_len:
                        full = full[..., -key_len:]
                    masks[layer_type] = mx.broadcast_to(
                        full[:, None, None, :], (B, 1, canvas_length, key_len)
                    )
                continue

            if decoder_attention_mask is None:
                window_prefix = max(self.config.sliding_window - 1, 0)
                if encoder_len == valid_encoder_len and encoder_len <= window_prefix:
                    masks[layer_type] = None
                    continue
                start = max(0, valid_encoder_len - window_prefix)
                positions = mx.arange(encoder_len)
                encoder_mask = (positions >= start) & (positions < valid_encoder_len)
                canvas_mask = mx.ones((canvas_length,), dtype=mx.bool_)
                row = mx.concatenate([encoder_mask, canvas_mask], axis=0)
                masks[layer_type] = mx.broadcast_to(
                    row[None, None, None, :], (B, 1, canvas_length, key_len)
                )
            else:
                full = decoder_attention_mask.astype(mx.bool_)
                if full.shape[-1] != key_len:
                    full = full[..., -key_len:]
                start = max(
                    0,
                    valid_encoder_len - max(self.config.sliding_window - 1, 0),
                )
                positions = mx.arange(encoder_len)
                keep = mx.concatenate(
                    [
                        (positions >= start) & (positions < valid_encoder_len),
                        mx.ones((canvas_length,), dtype=mx.bool_),
                    ],
                    axis=0,
                )
                row = full[:, None, None, :] & keep[None, None, None, :]
                masks[layer_type] = mx.broadcast_to(row, (B, 1, canvas_length, key_len))

        return masks

    def __call__(
        self,
        canvas_ids: mx.array,
        cache=None,
        self_conditioning_logits: Optional[mx.array] = None,
        self_conditioning_embeddings: Optional[mx.array] = None,
        decoder_attention_mask: Optional[mx.array] = None,
    ):
        h = self._embed_canvas(
            canvas_ids,
            self_conditioning_logits,
            self_conditioning_embeddings,
        )
        cache = cache or [None] * len(self.layers)
        masks = self._make_decoder_masks(h, cache, decoder_attention_mask)
        offset = _cache_offset(cache[0]) if cache else 0

        for layer, c in zip(self.layers, cache):
            h = layer(
                h,
                masks.get(layer.layer_type),
                c,
                decoder=True,
                offset=offset,
            )
        return self.norm(h)


class EncoderModel(nn.Module):
    def __init__(self, config: ModelConfig, decoder: DecoderModel):
        super().__init__()
        self.config = config
        self.text_config = config.text_config
        self.language_model = EncoderLanguageModel(decoder)
        # weakref.ref, not proxy: see EncoderLanguageModel.
        self._decoder_ref = weakref.ref(decoder)
        if config.vision_config is not None:
            self.vision_tower = VisionModel(config.vision_config)
            self.embed_vision = MultimodalEmbedder(
                embedding_dim=config.vision_config.hidden_size,
                text_hidden_size=config.text_config.hidden_size,
                eps=config.vision_config.rms_norm_eps,
            )
        else:
            self.vision_tower = None
            self.embed_vision = None

    @property
    def decoder(self):
        return self._decoder_ref()

    def make_cache(self, max_size: Optional[int] = None):
        caches = []
        for layer_type in self.text_config.layer_types:
            if max_size is not None:
                caches.append(StaticPrefixKVCache(max_size=max_size))
            elif layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.text_config.sliding_window))
        return caches

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache, draft_model, draft_kind
        prefill_kwargs = prefill_kwargs or {}
        if bool(prefill_kwargs.get("has_padding", False)):
            return False
        attention_mask = prefill_kwargs.get("attention_mask", None)
        if attention_mask is not None and not bool(mx.all(attention_mask).item()):
            return False
        if bool(prefill_kwargs.get("use_static_cache", False)):
            return False
        if prefill_kwargs.get("pixel_values", None) is not None:
            return False

        token_types = prefill_kwargs.get("mm_token_type_ids", None)
        if token_types is not None:
            # Visual spans can use bidirectional attention inside the whole
            # block. Keep those prompts on the single prefill path until
            # chunking can split on visual-block boundaries.
            has_visual = bool(mx.any((token_types == 1) | (token_types == 2)).item())
            if has_visual:
                return False

        return True

    def get_image_features(self, pixel_values):
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError(
                "This checkpoint does not include a vision tower; "
                "image inputs are not supported."
            )
        return self.embed_vision(self.vision_tower(pixel_values))

    def _embed_inputs(self, input_ids, pixel_values=None):
        image_mask = input_ids == self.config.image_token_id
        video_token_id = getattr(self.config, "video_token_id", None)
        video_mask = (
            input_ids == video_token_id
            if video_token_id is not None
            else mx.zeros_like(image_mask)
        )
        vision_mask = image_mask | video_mask
        llm_input_ids = mx.where(
            vision_mask,
            self.text_config.pad_token_id,
            input_ids,
        )
        inputs_embeds = (
            self.decoder.embed_tokens(llm_input_ids) * self.decoder.embed_scale
        )
        if pixel_values is not None:
            features = self.get_image_features(pixel_values).astype(inputs_embeds.dtype)
            mask_expanded = mx.broadcast_to(
                mx.expand_dims(vision_mask, -1), inputs_embeds.shape
            )
            inputs_embeds = masked_scatter(inputs_embeds, mask_expanded, features)
        return inputs_embeds

    def _vision_block_overlay(self, mm_token_type_ids, seq_len):
        """Bidirectional attention overlay for image-token blocks.

        Mirrors the reference encoder behavior when
        ``use_bidirectional_attention == "vision"``: tokens within the same
        contiguous image block attend to each other bidirectionally, on top of
        the usual causal (and sliding-window) mask.
        """
        if (
            getattr(self.text_config, "use_bidirectional_attention", None) != "vision"
            or mm_token_type_ids is None
            or seq_len <= 1
            or mm_token_type_ids.shape[-1] != seq_len
        ):
            return None
        is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        if not bool(mx.any(is_vision).item()):
            return None
        prev = mx.concatenate(
            [mx.zeros_like(is_vision[:, :1]), is_vision[:, :-1]], axis=1
        )
        starts = is_vision & ~prev
        group_ids = mx.cumsum(starts.astype(mx.int32), axis=1) - 1
        block_ids = mx.where(is_vision, group_ids, mx.zeros_like(group_ids) - 1)
        q_blocks = mx.expand_dims(block_ids, -1)
        k_blocks = mx.expand_dims(block_ids, -2)
        return (q_blocks != -1) & (q_blocks == k_blocks)

    def _make_encoder_masks(
        self, h, cache, attention_mask=None, mm_token_type_ids=None
    ):
        if isinstance(attention_mask, dict):
            return attention_mask

        B, N, _ = h.shape
        key_len = N + (_cache_offset(cache[0]) if cache else 0)
        overlay = self._vision_block_overlay(mm_token_type_ids, N)
        if overlay is not None and key_len != N:
            # Image blocks only appear in the prompt prefill, where the cache
            # is empty; ignore the overlay for continuation encoder passes.
            overlay = None

        if attention_mask is None and overlay is None:
            return [
                create_attention_mask(
                    h,
                    c,
                    window_size=(
                        self.text_config.sliding_window
                        if layer.layer_type == "sliding_attention"
                        else None
                    ),
                )
                for layer, c in zip(self.decoder.layers, cache)
            ]

        if attention_mask is None:
            key_mask = mx.ones((B, key_len), dtype=mx.bool_)
        else:
            key_mask = attention_mask.astype(mx.bool_)
            if key_mask.shape[-1] != key_len:
                key_mask = key_mask[..., -key_len:]
        positions = mx.arange(key_len)
        q_positions = mx.arange(key_len - N, key_len)[:, None]
        base = q_positions >= positions[None, :]
        masks = []
        for layer in self.decoder.layers:
            m = base
            if layer.layer_type == "sliding_attention":
                m = m & (
                    q_positions < positions[None, :] + self.text_config.sliding_window
                )
            m = m[None, None, :, :]
            if overlay is not None:
                m = m | overlay[:, None, :, :]
            m = m & key_mask[:, None, None, :]
            masks.append(mx.broadcast_to(m, (B, 1, N, key_len)))
        return masks

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache=None,
        pixel_values: Optional[mx.array] = None,
        mm_token_type_ids: Optional[mx.array] = None,
    ):
        h = self._embed_inputs(
            input_ids,
            pixel_values=pixel_values,
        )
        if cache is None:
            cache = self.make_cache()
        masks = self._make_encoder_masks(
            h, cache, attention_mask, mm_token_type_ids=mm_token_type_ids
        )

        for i, (layer, c, mask) in enumerate(zip(self.decoder.layers, cache, masks)):
            h = layer(
                h,
                mask,
                c,
                decoder=False,
                layer_scalar=self.language_model.layers[i].layer_scalar,
            )
        return self.decoder.norm(h), cache


class DiffusionGemma4Backbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.decoder = DecoderModel(config.text_config)
        self.encoder = EncoderModel(config, self.decoder)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache=None,
        canvas_ids: Optional[mx.array] = None,
        self_conditioning_logits: Optional[mx.array] = None,
        self_conditioning_embeddings: Optional[mx.array] = None,
        decoder_attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mm_token_type_ids: Optional[mx.array] = None,
    ):
        if input_ids is not None:
            _, cache = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                cache=cache,
                pixel_values=pixel_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        elif cache is None:
            raise ValueError("Either input_ids or cache must be provided.")

        if canvas_ids is None:
            batch_size = input_ids.shape[0]
            canvas_ids = mx.random.randint(
                0,
                self.config.text_config.vocab_size,
                (batch_size, self.config.canvas_length),
            )

        hidden_states = self.decoder(
            canvas_ids,
            cache=cache,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_embeddings=self_conditioning_embeddings,
            decoder_attention_mask=decoder_attention_mask,
        )
        return hidden_states, cache


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig(text_config=args)
        self.model_type = args.model_type
        self.model = DiffusionGemma4Backbone(self.config)
        self.final_logit_softcapping = args.final_logit_softcapping
        self._softcap = make_compiled_softcap(float(args.final_logit_softcapping))

    @property
    def layers(self):
        return self.model.decoder.layers

    def make_cache(self, max_size: Optional[int] = None):
        return self.model.encoder.make_cache(max_size=max_size)

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del input_ids, inputs_embeds, prompt_cache, draft_model, draft_kind
        return self.model.encoder.chunked_prefill_policy(prefill_kwargs=prefill_kwargs)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        cache=None,
        canvas_ids: Optional[mx.array] = None,
        self_conditioning_logits: Optional[mx.array] = None,
        self_conditioning_embeddings: Optional[mx.array] = None,
        decoder_attention_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        hidden_states, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            canvas_ids=canvas_ids,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_embeddings=self_conditioning_embeddings,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = self.model.decoder.embed_tokens.as_linear(hidden_states)
        logits = self._softcap(logits)
        return LanguageModelOutput(logits=logits, hidden_states=[hidden_states])

    @property
    def quant_predicate(self):
        return diffusion_gemma_quant_predicate
