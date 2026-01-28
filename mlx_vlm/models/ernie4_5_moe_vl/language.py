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
    """Rotary Position Embedding for ERNIE 4.5 VL with MRoPE support."""

    def __init__(
        self,
        dim: int,
        base: float = 10000,
        freq_allocation: int = 20,
    ):
        self.dim = dim  # head_dim
        self.base = base
        self.freq_allocation = freq_allocation  # frequencies allocated to temporal dim

        # Pre-compute inverse frequencies for all positions
        indices = mx.arange(0, self.dim, 2, dtype=mx.float32)
        self.inv_freq = 1.0 / (self.base ** (indices / self.dim))

    def _compute_pos_emb(self, position_ids):
        """Compute sin/cos embeddings for given position IDs.

        Args:
            position_ids: [batch, seq_len] or scalar
        Returns:
            pos_emb: [batch, seq_len, head_dim] with sin in first half, cos in second half
        """
        # position_ids: [batch, seq_len] -> [batch, seq_len, 1]
        if position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        position_ids = position_ids.astype(mx.float32)

        # sinusoid_inp: [batch, seq_len, head_dim//2]
        sinusoid_inp = position_ids[:, :, None] * self.inv_freq[None, None, :]

        # pos_emb: [batch, seq_len, head_dim] - sin first, cos second
        pos_emb = mx.concatenate([mx.sin(sinusoid_inp), mx.cos(sinusoid_inp)], axis=-1)
        return pos_emb

    def __call__(self, x, position_ids):
        """
        Compute 3D rotary embeddings matching PyTorch's apply_rotary_3d.

        Args:
            x: Input tensor for dtype reference
            position_ids: Position IDs, shape (batch, seq_len, 3) for 3D positions [T, H, W]

        Returns:
            sin_pos, cos_pos: [batch, seq_len, head_dim] ready for rotation
        """
        if position_ids.ndim == 2:
            # 1D positions - expand to 3D with same values
            position_ids = mx.stack([position_ids, position_ids, position_ids], axis=-1)

        batch_size, seq_len, _ = position_ids.shape
        half_dim = self.dim // 2
        freq_alloc = self.freq_allocation

        # Compute full position embeddings for maximum position
        max_pos = int(mx.max(position_ids).item()) + 1
        full_positions = mx.arange(max_pos)
        full_emb = self._compute_pos_emb(
            full_positions[None, :]
        )  # [1, max_pos, head_dim]

        # Split into sin and cos (each head_dim//2)
        full_sin = full_emb[0, :, :half_dim]  # [max_pos, head_dim//2]
        full_cos = full_emb[0, :, half_dim:]  # [max_pos, head_dim//2]

        # Extract positions for each dimension
        pos_t = position_ids[:, :, 0]  # [batch, seq_len] temporal
        pos_h = position_ids[:, :, 1]  # [batch, seq_len] height
        pos_w = position_ids[:, :, 2]  # [batch, seq_len] width

        # Gather sin/cos for each position
        # sin_t: temporal uses last freq_allocation frequencies
        sin_t = full_sin[pos_t, -freq_alloc:]  # [batch, seq_len, freq_alloc]
        cos_t = full_cos[pos_t, -freq_alloc:]

        # sin_h: height uses even indices of first (half_dim - freq_alloc)
        hw_range = half_dim - freq_alloc  # 44 for dim=128, freq=20
        sin_h = full_sin[pos_h, :hw_range:2]  # [batch, seq_len, hw_range//2]
        cos_h = full_cos[pos_h, :hw_range:2]

        # sin_w: width uses odd indices of first (half_dim - freq_alloc)
        sin_w = full_sin[pos_w, 1:hw_range:2]  # [batch, seq_len, hw_range//2]
        cos_w = full_cos[pos_w, 1:hw_range:2]

        # Interleave H and W: [h0, w0, h1, w1, ...]
        sin_hw = mx.stack([sin_h, sin_w], axis=-1).reshape(
            batch_size, seq_len, -1
        )  # [batch, seq_len, hw_range]
        cos_hw = mx.stack([cos_h, cos_w], axis=-1).reshape(batch_size, seq_len, -1)

        # Concatenate HW and T
        sin_thw = mx.concatenate([sin_hw, sin_t], axis=-1)  # [batch, seq_len, half_dim]
        cos_thw = mx.concatenate([cos_hw, cos_t], axis=-1)

        # Double for full head_dim: [s0, s0, s1, s1, ...]
        sin_pos = mx.stack([sin_thw, sin_thw], axis=-1).reshape(
            batch_size, seq_len, self.dim
        )
        cos_pos = mx.stack([cos_thw, cos_thw], axis=-1).reshape(
            batch_size, seq_len, self.dim
        )

        return cos_pos.astype(x.dtype), sin_pos.astype(x.dtype)


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

        # Get freq_allocation for 3D RoPE (temporal dimension frequency allocation)
        self.freq_allocation = getattr(args, "freq_allocation", 20)

        self.rotary_emb = Ernie4_5RotaryEmbedding(
            head_dim,
            base=args.rope_theta,
            freq_allocation=self.freq_allocation,
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

        # Apply rotary embeddings
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
    """Standard MLP for ERNIE."""

    def __init__(self, dim, hidden_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Ernie4_5_MoeMLP(nn.Module):
    """Mixture of Experts MLP for ERNIE with dual expert groups (text + multimodal)."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.k = args.moe_k

        # Parse expert configuration - ERNIE has two groups of experts
        moe_num_experts = args.moe_num_experts
        moe_intermediate_size = args.moe_intermediate_size

        if isinstance(moe_num_experts, (list, tuple)) and len(moe_num_experts) == 2:
            # Two groups: text experts and multimodal experts
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

        # Text experts gate and switch_mlp
        self.gate = nn.Linear(args.hidden_size, self.num_text_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            self.text_intermediate_size,
            self.num_text_experts,
            bias=args.use_bias,
        )

        # Multimodal experts gate and switch_mlp (if present)
        if self.has_dual_experts and self.num_mm_experts > 0:
            self.gate_1 = nn.Linear(args.hidden_size, self.num_mm_experts, bias=False)
            self.switch_mlp_1 = SwitchGLU(
                args.hidden_size,
                self.mm_intermediate_size,
                self.num_mm_experts,
                bias=args.use_bias,
            )

        # Shared experts
        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                self.text_intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_MLP(
                args.hidden_size, shared_intermediate_size, args.use_bias
            )
        else:
            self.shared_experts = None

        self.gate_act = nn.Softmax() if args.moe_gate_act == "softmax" else nn.Sigmoid()
        self.norm_gate_logits = getattr(args, "moe_norm_gate_logits", True)

    def __call__(
        self, x: mx.array, token_type_ids: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass through the MoE layer.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            token_type_ids: Optional token type IDs to route text vs multimodal tokens.
                            0 = text tokens -> text experts, >0 = multimodal tokens -> MM experts.

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        k = self.k

        # Process text experts (gate + switch_mlp)
        gates = self.gate(x)
        gates = self.gate_act(gates.astype(mx.float32))
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        if self.norm_gate_logits:
            scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y_text = self.switch_mlp(x, inds)
        y_text = (y_text * scores[..., None]).sum(axis=-2).astype(y_text.dtype)

        # Route based on token_type_ids
        if (
            not self.has_dual_experts
            or self.num_mm_experts == 0
            or token_type_ids is None
        ):
            # Text-only: use only text experts
            y = y_text
        else:
            # Multimodal: process MM experts and selectively route
            gates_mm = self.gate_1(x)
            gates_mm = self.gate_act(gates_mm.astype(mx.float32))
            inds_mm = mx.stop_gradient(
                mx.argpartition(-gates_mm, kth=k - 1, axis=-1)[..., :k]
            )
            scores_mm = mx.take_along_axis(gates_mm, inds_mm, axis=-1)

            if self.norm_gate_logits:
                scores_mm = scores_mm / mx.maximum(
                    scores_mm.sum(axis=-1, keepdims=True), 1e-12
                )

            y_mm = self.switch_mlp_1(x, inds_mm)
            y_mm = (y_mm * scores_mm[..., None]).sum(axis=-2).astype(y_mm.dtype)

            # Select based on token type: text tokens -> text experts, mm tokens -> mm experts
            is_text = token_type_ids == 0  # [batch, seq_len]
            is_text_expanded = mx.expand_dims(is_text, axis=-1)  # [batch, seq_len, 1]
            y = mx.where(is_text_expanded, y_text, y_mm)

        # Add shared experts output
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Ernie4_5VLDecoderLayer(nn.Module):
    """Decoder layer for ERNIE 4.5 VL."""

    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)

        # Determine MoE layer boundaries
        moe_layer_start_index = args.moe_layer_start_index
        if isinstance(moe_layer_start_index, (tuple, list)):
            moe_layer_start_index = min(moe_layer_start_index)

        moe_layer_end_index = args.moe_layer_end_index
        if moe_layer_end_index is None:
            moe_layer_end_index = args.num_hidden_layers - 1
        elif isinstance(moe_layer_end_index, (tuple, list)):
            moe_layer_end_index = max(moe_layer_end_index)

        # Use MoE if within the MoE layer range
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
    """ERNIE 4.5 transformer model."""

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
    """Language model wrapper for ERNIE 4.5 VL."""

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
        """Calculate 3D RoPE index for image/video tokens."""
        batch_size, seq_length = input_ids.shape
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if image_grid_thw is not None or video_grid_thw is not None:
            # Build position_ids for each batch element
            batch_position_ids = []
            mrope_position_deltas = []

            image_index, video_index = 0, 0

            for i in range(batch_size):
                input_tokens = input_ids[i].tolist()
                llm_pos_ids_list = []
                st = 0

                # Count images and videos by looking at vision_start tokens
                # This is more robust than counting all image/video tokens
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
                    else:
                        t, h, w = video_grid_thw[video_index].tolist()
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                    # Text positions (same for all 3 dims)
                    text_pos = mx.arange(text_len) + st_idx
                    text_pos_3d = mx.stack([text_pos, text_pos, text_pos], axis=0)
                    llm_pos_ids_list.append(text_pos_3d)

                    # Image/video positions
                    t_idx = mx.repeat(
                        mx.arange(llm_grid_t).reshape(-1, 1),
                        llm_grid_h * llm_grid_w,
                        axis=1,
                    ).flatten()
                    h_idx = mx.tile(
                        mx.arange(llm_grid_h).reshape(1, -1, 1),
                        (llm_grid_t, 1, llm_grid_w),
                    ).flatten()
                    w_idx = mx.tile(
                        mx.arange(llm_grid_w).reshape(1, 1, -1),
                        (llm_grid_t, llm_grid_h, 1),
                    ).flatten()

                    vision_pos = (
                        mx.stack([t_idx, h_idx, w_idx], axis=0) + text_len + st_idx
                    )
                    llm_pos_ids_list.append(vision_pos)

                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

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

            # Stack all batch position IDs
            position_ids = mx.stack(
                batch_position_ids, axis=0
            )  # [batch_size, seq_len, 3]
            mrope_position_deltas = mx.array(mrope_position_deltas)
            return position_ids, mrope_position_deltas
        else:
            # Standard sequential positions
            position_ids = mx.arange(seq_length)
            position_ids = mx.broadcast_to(
                position_ids[None, :], (batch_size, seq_length)
            )
            # Expand to 3D format
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

        # Reset rope deltas when processing new image/video
        if pixel_values is not None:
            self._rope_deltas = None

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            cache_offset = offset.item() if isinstance(offset, mx.array) else offset

        if position_ids is None and (mask is None or mask.ndim == 2):
            # Calculate 3D RoPE positions
            if (
                cache is None or cache[0] is None or cache_offset == 0
            ) or self._rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, mask
                )
                self._rope_deltas = rope_deltas
            else:
                # Use cached rope deltas for continuation
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
            "e_score_correction_bias",
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

        # Remove lm_head if tie_word_embeddings is True
        if self.args.tie_word_embeddings:
            lm_head_key = "language_model.lm_head.weight"
            if lm_head_key in weights:
                weights.pop(lm_head_key)

        return weights
