"""Language model for ERNIE 4.5 VL MoE."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig


class Ernie4_5RotaryEmbedding:
    """Rotary Position Embedding for ERNIE 4.5 VL with MRoPE support.

    Matches PyTorch's implementation with pre-rotated inverse frequencies.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000,
        mrope_section: tuple = (22, 22, 20),
    ):
        self.dim = dim  # head_dim
        self.base = base
        self.mrope_section = mrope_section  # (h_dim, w_dim, t_dim)

        # Pre-compute inverse frequencies
        indices = mx.arange(0, self.dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.base ** (indices / self.dim))

        # Pre-rotate frequencies to match PyTorch's approach
        # This avoids rotation during forward pass
        hw_dim = mrope_section[0] + mrope_section[1]  # 44
        t_dim = mrope_section[2]  # 20

        inv_freq_3d = mx.zeros_like(inv_freq)
        # Pre-rotate HW dimensions: [even, odd] -> interleaved during recomposition
        hw_freqs = inv_freq[:-t_dim]  # First (dim/2 - t_dim) frequencies
        inv_freq_3d = mx.concatenate(
            [
                mx.concatenate([hw_freqs[0::2], hw_freqs[1::2]]),  # Pre-rotated HW
                inv_freq[-t_dim:],  # T frequencies unchanged
            ]
        )
        self.inv_freq = inv_freq_3d

    def _recomposition_to_3d(self, freq):
        """Recompose frequencies for 3D positions matching PyTorch's approach.

        Args:
            freq: [3, batch, seq_len, dim//2] - frequencies for T, H, W dimensions

        Returns:
            Recomposed frequencies [batch, seq_len, dim]
        """
        # Split by mrope_section
        h_dim, w_dim, t_dim = self.mrope_section

        # freq shape: [3, batch, seq_len, half_dim]
        # Split each dimension's frequencies
        freq_parts = []
        for i in range(3):
            freq_parts.append(mx.split(freq[i], [h_dim, h_dim + w_dim], axis=-1))

        # Recompose: freq_h from dim 1, freq_w from dim 2, freq_t from dim 0
        # This matches PyTorch's (i + 1) % 3 indexing
        freq_h = freq_parts[1][0]  # H from position 1
        freq_w = freq_parts[2][1]  # W from position 2
        freq_t = freq_parts[0][2]  # T from position 0

        # Interleave H and W: [h0, w0, h1, w1, ...]
        freq_hw = mx.stack([freq_h, freq_w], axis=-1).reshape(
            freq_h.shape[0], freq_h.shape[1], -1
        )

        # Concatenate HW and T
        freq_hwt = mx.concatenate([freq_hw, freq_t], axis=-1)

        # Repeat interleave by 2 for full head_dim
        freq_full = mx.repeat(freq_hwt, 2, axis=-1)

        return freq_full

    def __call__(self, x, position_ids):
        """
        Compute 3D rotary embeddings matching PyTorch's implementation.

        Args:
            x: Input tensor for dtype reference
            position_ids: Position IDs, shape (batch, seq_len, 3) for 3D positions [T, H, W]

        Returns:
            cos, sin: [batch, seq_len, head_dim] ready for rotation
        """
        if position_ids.ndim == 2:
            # 1D positions - expand to 3D with same values
            position_ids = mx.stack([position_ids, position_ids, position_ids], axis=-1)

        batch_size, seq_len, _ = position_ids.shape

        # position_ids: [batch, seq_len, 3] -> [3, batch, seq_len]
        position_ids = position_ids.transpose(2, 0, 1).astype(mx.float32)

        # inv_freq: [dim//2] -> [1, 1, dim//2, 1] for broadcasting
        inv_freq_expanded = self.inv_freq[None, None, :, None]  # [1, 1, dim//2, 1]
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded, (3, batch_size, self.dim // 2, 1)
        )

        # position_ids: [3, batch, seq_len] -> [3, batch, 1, seq_len]
        position_ids_expanded = position_ids[:, :, None, :]

        # freqs: [3, batch, dim//2, seq_len] -> [3, batch, seq_len, dim//2]
        freqs = (inv_freq_expanded * position_ids_expanded).transpose(0, 1, 3, 2)

        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        # Recompose to 3D
        cos = self._recomposition_to_3d(cos)
        sin = self._recomposition_to_3d(sin)

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half_interleaved(x):
    """Rotates using interleaved pattern: [-x1, x0, -x3, x2, ...].

    This matches PyTorch's rotation: stack([-x[1::2], x[0::2]], dim=-1).reshape()
    """
    x_even = x[..., 0::2]  # [x0, x2, x4, ...]
    x_odd = x[..., 1::2]  # [x1, x3, x5, ...]
    # Stack as [-odd, even] and reshape
    rotated = mx.stack([-x_odd, x_even], axis=-1)
    return rotated.reshape(x.shape)


def apply_rotary_pos_emb(q, k, cos_pos, sin_pos):
    """Apply rotary position embeddings to queries and keys.

    Uses interleaved rotation matching PyTorch's apply_rotary_3d.

    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_kv_heads, seq_len, head_dim]
        cos_pos: [batch, seq_len, head_dim]
        sin_pos: [batch, seq_len, head_dim]
    """
    orig_dtype = q.dtype
    # Expand for heads dimension

    cos_pos = mx.expand_dims(cos_pos, axis=1)  # [batch, 1, seq_len, head_dim]
    sin_pos = mx.expand_dims(sin_pos, axis=1)

    # Apply rotation: q_rotated = q * cos + rotate_half(q) * sin
    q_rotated = rotate_half_interleaved(q)
    k_rotated = rotate_half_interleaved(k)

    q_embed = (q.astype(mx.float32) * cos_pos) + (
        q_rotated.astype(mx.float32) * sin_pos
    )
    k_embed = (k.astype(mx.float32) * cos_pos) + (
        k_rotated.astype(mx.float32) * sin_pos
    )

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class Attention(nn.Module):
    """Multi-headed attention for ERNIE 4.5 with MRoPE support."""

    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads or n_heads

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.use_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.use_bias)

        # Get mrope_section for 3D RoPE (H, W, T dimension allocation)
        # Default [22, 22, 20] for head_dim=128
        self.mrope_section = tuple(getattr(args, "mrope_section", [22, 22, 20]))

        self.rotary_emb = Ernie4_5RotaryEmbedding(
            head_dim,
            base=args.rope_theta,
            mrope_section=self.mrope_section,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape and transpose: [B, L, n_heads, head_dim] -> [B, n_heads, L, head_dim]
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Handle position IDs
        if position_ids is None:
            offset = cache.offset if cache is not None else 0
            position_ids = mx.arange(offset, offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)

        cos, sin = self.rotary_emb(values, position_ids)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Ernie4_5_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Ernie4_5_MoeMLP(nn.Module):
    """Mixture of Experts MLP for ERNIE with dual expert groups."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.k = args.moe_k
        self.norm_min = getattr(args, "moe_norm_min", 1e-12)

        moe_num_experts = args.moe_num_experts
        moe_intermediate_size = args.moe_intermediate_size

        if isinstance(moe_num_experts, (list, tuple)) and len(moe_num_experts) == 2:
            self.num_text_experts = moe_num_experts[0]
            self.num_mm_experts = moe_num_experts[1]
            self.has_dual_experts = True
        else:
            self.num_text_experts = (
                moe_num_experts
                if not isinstance(moe_num_experts, (list, tuple))
                else moe_num_experts[0]
            )
            self.num_mm_experts = 0
            self.has_dual_experts = False

        if (
            isinstance(moe_intermediate_size, (list, tuple))
            and len(moe_intermediate_size) == 2
        ):
            self.text_intermediate_size = moe_intermediate_size[0]
            self.mm_intermediate_size = moe_intermediate_size[1]
        else:
            self.text_intermediate_size = (
                moe_intermediate_size
                if not isinstance(moe_intermediate_size, (list, tuple))
                else moe_intermediate_size[0]
            )
            self.mm_intermediate_size = self.text_intermediate_size

        self.gate = nn.Linear(args.hidden_size, self.num_text_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((self.num_text_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            self.text_intermediate_size,
            self.num_text_experts,
            bias=args.use_bias,
        )

        if self.has_dual_experts and self.num_mm_experts > 0:
            self.gate_1 = nn.Linear(args.hidden_size, self.num_mm_experts, bias=False)
            self.e_score_correction_bias_1 = mx.zeros((self.num_mm_experts,))
            self.switch_mlp_1 = SwitchGLU(
                args.hidden_size,
                self.mm_intermediate_size,
                self.num_mm_experts,
                bias=args.use_bias,
            )

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                self.text_intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_MLP(
                args.hidden_size, shared_intermediate_size, args.use_bias
            )
        else:
            self.shared_experts = None

    def _route_experts(
        self, x: mx.array, gate: nn.Module, e_score_correction_bias: mx.array
    ) -> tuple:
        k = self.k
        router_logits = gate(x).astype(mx.float32)
        routing_weights = mx.softmax(router_logits, axis=-1)
        routing_weights_with_bias = routing_weights + e_score_correction_bias

        selected_experts = mx.stop_gradient(
            mx.argpartition(-routing_weights_with_bias, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(routing_weights, selected_experts, axis=-1)
        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), self.norm_min)

        return selected_experts, scores

    def __call__(
        self, x: mx.array, token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        inds, scores = self._route_experts(x, self.gate, self.e_score_correction_bias)
        y_text = self.switch_mlp(x, inds)
        y_text = (y_text * scores[..., None]).sum(axis=-2).astype(y_text.dtype)

        if (
            not self.has_dual_experts
            or self.num_mm_experts == 0
            or token_type_ids is None
        ):
            y = y_text
        else:
            inds_mm, scores_mm = self._route_experts(
                x, self.gate_1, self.e_score_correction_bias_1
            )
            y_mm = self.switch_mlp_1(x, inds_mm)
            y_mm = (y_mm * scores_mm[..., None]).sum(axis=-2).astype(y_mm.dtype)

            is_text = token_type_ids == 0
            is_text_expanded = mx.expand_dims(is_text, axis=-1)
            y = mx.where(is_text_expanded, y_text, y_mm)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Ernie4_5VLDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)

        moe_layer_start_index = args.moe_layer_start_index
        if isinstance(moe_layer_start_index, (tuple, list)):
            moe_layer_start_index = min(moe_layer_start_index)

        moe_layer_end_index = args.moe_layer_end_index
        if moe_layer_end_index is None:
            moe_layer_end_index = args.num_hidden_layers - 1
        elif isinstance(moe_layer_end_index, (tuple, list)):
            moe_layer_end_index = max(moe_layer_end_index)

        use_moe = (
            ((layer_idx + 1) % args.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        )

        if use_moe:
            self.mlp = Ernie4_5_MoeMLP(args)
        else:
            self.mlp = Ernie4_5_MLP(
                args.hidden_size, args.intermediate_size, args.use_bias
            )

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        if isinstance(self.mlp, Ernie4_5_MoeMLP):
            r = self.mlp(
                self.post_attention_layernorm(h), token_type_ids=token_type_ids
            )
        else:
            r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Ernie4_5Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Ernie4_5VLDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids, token_type_ids=token_type_ids)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Ernie4_5Model(args)
        self._rope_deltas = None

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        batch_size, seq_length = input_ids.shape
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if image_grid_thw is not None or video_grid_thw is not None:
            batch_position_ids = []
            mrope_position_deltas = []

            image_index, video_index = 0, 0

            for i in range(batch_size):
                input_tokens = input_ids[i].tolist()
                llm_pos_ids_list = []
                st = 0

                image_nums, video_nums = 0, 0
                for idx, token in enumerate(input_tokens):
                    if token == vision_start_token_id and idx + 1 < len(input_tokens):
                        next_token = input_tokens[idx + 1]
                        if next_token == image_token_id:
                            image_nums += 1
                        elif next_token == video_token_id:
                            video_nums += 1

                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if image_token_id in input_tokens[st:] and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if video_token_id in input_tokens[st:] and remain_videos > 0
                        else len(input_tokens) + 1
                    )

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index].tolist()
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                        vision_token = image_token_id
                    else:
                        t, h, w = video_grid_thw[video_index].tolist()
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                        vision_token = video_token_id

                    llm_grid_t = t
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    expected_vision_len = llm_grid_t * llm_grid_h * llm_grid_w

                    actual_vision_len = 0
                    for j in range(
                        ed, min(ed + expected_vision_len, len(input_tokens))
                    ):
                        if input_tokens[j] == vision_token:
                            actual_vision_len += 1
                        else:
                            break

                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                    text_pos = mx.arange(text_len) + st_idx
                    text_pos_3d = mx.stack([text_pos, text_pos, text_pos], axis=0)
                    llm_pos_ids_list.append(text_pos_3d)

                    if actual_vision_len > 0:
                        t_idx = mx.repeat(
                            mx.arange(llm_grid_t).reshape(-1, 1),
                            llm_grid_h * llm_grid_w,
                            axis=1,
                        ).flatten()[:actual_vision_len]
                        h_idx = mx.tile(
                            mx.arange(llm_grid_h).reshape(1, -1, 1),
                            (llm_grid_t, 1, llm_grid_w),
                        ).flatten()[:actual_vision_len]
                        w_idx = mx.tile(
                            mx.arange(llm_grid_w).reshape(1, 1, -1),
                            (llm_grid_t, llm_grid_h, 1),
                        ).flatten()[:actual_vision_len]

                        vision_pos = (
                            mx.stack([t_idx, h_idx, w_idx], axis=0) + text_len + st_idx
                        )
                        llm_pos_ids_list.append(vision_pos)

                    st = ed + actual_vision_len

                # Handle remaining text
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    text_pos = mx.arange(text_len) + st_idx
                    text_pos_3d = mx.stack([text_pos, text_pos, text_pos], axis=0)
                    llm_pos_ids_list.append(text_pos_3d)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1)  # [3, seq_len]
                batch_position_ids.append(llm_positions.T)  # [seq_len, 3]
                mrope_position_deltas.append(llm_positions.max() + 1 - seq_length)

            position_ids = mx.stack(batch_position_ids, axis=0)
            mrope_position_deltas = mx.array(mrope_position_deltas)
            return position_ids, mrope_position_deltas
        else:
            position_ids = mx.arange(seq_length)
            position_ids = mx.broadcast_to(
                position_ids[None, :], (batch_size, seq_length)
            )
            position_ids = mx.stack([position_ids, position_ids, position_ids], axis=-1)
            return position_ids, mx.zeros((batch_size,), dtype=mx.int32)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values is not None:
            self._rope_deltas = None

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            cache_offset = offset.item() if isinstance(offset, mx.array) else offset

        if position_ids is None and (mask is None or mask.ndim == 2):
            if (
                cache is None or cache[0] is None or cache_offset == 0
            ) or self._rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, mask
                )
                self._rope_deltas = rope_deltas
            else:
                batch_size, seq_length = inputs.shape
                delta = cache_offset + self._rope_deltas if cache is not None else 0
                position_ids = mx.arange(seq_length) + delta
                position_ids = mx.broadcast_to(
                    position_ids[None, :], (batch_size, seq_length)
                )
                position_ids = mx.stack(
                    [position_ids, position_ids, position_ids], axis=-1
                )

        token_type_ids = kwargs.pop("token_type_ids", None)

        out = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mask=mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return LanguageModelOutput(logits=out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def sanitize(self, weights):
        """Sanitize weights for loading."""
        remove_patterns = [
            "mtp_block.",
            "mtp_linear_proj.",
            "mtp_hidden_norm.",
            "mtp_emb_norm.",
        ]

        weights = {
            key: value
            for key, value in weights.items()
            if not any(pattern in key for pattern in remove_patterns)
        }

        # Get expert configuration
        moe_num_experts = self.args.moe_num_experts
        if isinstance(moe_num_experts, (list, tuple)) and len(moe_num_experts) == 2:
            num_text_experts = moe_num_experts[0]
            num_mm_experts = moe_num_experts[1]
        else:
            num_text_experts = (
                moe_num_experts
                if not isinstance(moe_num_experts, (list, tuple))
                else moe_num_experts[0]
            )
            num_mm_experts = 0

        for l in range(self.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}"

            # Stack text experts (0 to num_text_experts-1) into switch_mlp
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(num_text_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            # Stack multimodal experts (num_text_experts to num_text_experts+num_mm_experts-1) into switch_mlp_1
            if num_mm_experts > 0:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        first_mm_expert = num_text_experts
                        if f"{prefix}.mlp.experts.{first_mm_expert}.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(
                                    num_text_experts, num_text_experts + num_mm_experts
                                )
                            ]
                            weights[f"{prefix}.mlp.switch_mlp_1.{m}.{k}"] = mx.stack(
                                to_join
                            )

            # Transpose gate weights if needed (HuggingFace uses [in, out], MLX uses [out, in])
            # MLX nn.Linear(in=2560, out=64) expects shape (64, 2560), HF provides (2560, 64)
            gate_key = f"{prefix}.mlp.gate.weight"
            if gate_key in weights:
                w = weights[gate_key]
                # Only transpose if shape is (hidden_size, num_experts) not (num_experts, hidden_size)
                if w.shape[0] > w.shape[1]:  # (2560, 64) needs transpose
                    weights[gate_key] = w.T

            # Rename gate.weight_1 to gate_1.weight for multimodal gate and transpose
            gate_1_key = f"{prefix}.mlp.gate.weight_1"
            if gate_1_key in weights:
                w = weights.pop(gate_1_key)
                if w.shape[0] > w.shape[1]:  # Only transpose if needed
                    w = w.T
                weights[f"{prefix}.mlp.gate_1.weight"] = w

            # Handle e_score_correction_bias
            # HuggingFace stores as [2, num_experts] - row 0 for text, row 1 for multimodal
            bias_key = f"{prefix}.mlp.moe_statics.e_score_correction_bias"
            if bias_key in weights:
                bias = weights.pop(bias_key)
                if bias.ndim == 2 and bias.shape[0] == 2:
                    # Split into text and multimodal biases
                    weights[f"{prefix}.mlp.e_score_correction_bias"] = bias[0]
                    if num_mm_experts > 0:
                        weights[f"{prefix}.mlp.e_score_correction_bias_1"] = bias[1]
                else:
                    # Single bias (squeeze if needed)
                    if bias.ndim > 1:
                        bias = bias.squeeze()
                    weights[f"{prefix}.mlp.e_score_correction_bias"] = bias

        # Remove lm_head if tie_word_embeddings is True
        if self.args.tie_word_embeddings:
            lm_head_key = "language_model.lm_head.weight"
            if lm_head_key in weights:
                weights.pop(lm_head_key)

        return weights
