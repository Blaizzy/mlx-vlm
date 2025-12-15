"""Language model for ERNIE 4.5 VL MoE."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
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
        mrope_section: list = None,
    ):
        self.dim = dim
        self.base = base
        self.mrope_section = mrope_section or [36, 36, 20]  # Default for ERNIE

        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )
        self.inv_freq = inv_freq

    def apply_mrope(self, freqs, mrope_section):
        """Apply MRoPE to 3D rotary embeddings."""
        # freqs: (3, bs, seq_len, head_dim // 2)
        # Combine T, H, W frequencies according to mrope_section
        freqs_t = freqs[0]  # Start with T dimension (will overwrite parts)
        offset = mrope_section[0]
        for dim_idx, length in enumerate(mrope_section[1:], start=1):
            # Replace the slice with frequencies from H or W dimension
            freqs_t = mx.concatenate(
                [
                    freqs_t[..., :offset],
                    freqs[dim_idx, ..., offset : offset + length],
                    freqs_t[..., offset + length :],
                ],
                axis=-1,
            )
            offset += length
        return freqs_t

    def __call__(self, x, position_ids):
        """
        Compute rotary embeddings.

        Args:
            x: Input tensor for dtype reference
            position_ids: Position IDs, shape (batch, seq_len, 3) for 3D or (batch, seq_len) for 1D
        """
        # Handle 3D position IDs
        if position_ids.ndim == 3:
            # position_ids: (batch, seq_len, 3) -> (3, batch, seq_len)
            position_ids = position_ids.transpose(2, 0, 1)
        else:
            # Broadcast 1D to 3D format
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        # Expand inv_freq: (3, batch, head_dim//2, 1)
        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )

        # Position IDs: (3, batch, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        # Compute frequencies: (3, batch, head_dim//2, seq_len) -> (3, batch, seq_len, head_dim//2)
        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)

        # Apply MRoPE
        freqs = self.apply_mrope(freqs, self.mrope_section)

        # Create full embeddings
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys."""
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


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

        # MRoPE section based on freq_allocation
        freq_alloc = getattr(args, "freq_allocation", 20)
        half_dim = head_dim // 2
        spatial_dim = (half_dim - freq_alloc) // 2
        self.mrope_section = [spatial_dim, spatial_dim, freq_alloc]

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
            token_type_ids: Optional token type IDs to route text vs multimodal tokens

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        k = self.k

        # Process text experts
        gates = self.gate(x)
        gates = self.gate_act(gates.astype(mx.float32))
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        if self.norm_gate_logits:
            scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        # Process multimodal experts if present
        if self.has_dual_experts and self.num_mm_experts > 0:
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

            # Combine text and multimodal expert outputs
            # Simple approach: average both outputs
            # TODO: Could use token_type_ids to selectively route
            y = y + y_mm

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
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
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
            h = layer(h, mask, c, position_ids)

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
            # Initialize position_ids with shape [batch_size, seq_length, 3]
            position_ids = mx.zeros((batch_size, seq_length, 3), dtype=mx.int32)
            mrope_position_deltas = []

            image_index, video_index = 0, 0

            for i in range(batch_size):
                input_tokens = input_ids[i].tolist()
                llm_pos_ids_list = []
                st = 0

                # Count images and videos
                image_nums = input_tokens.count(image_token_id)
                video_nums = input_tokens.count(video_token_id)
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
                position_ids = position_ids.at[i].set(llm_positions.T)  # [seq_len, 3]
                mrope_position_deltas.append(llm_positions.max() + 1 - seq_length)

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

        out = self.model(
            inputs, cache=cache, inputs_embeds=inputs_embeds, position_ids=position_ids
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
