from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .config import ModelConfig, TextConfig


def _compute_default_rope_parameters(
    config: Optional[TextConfig] = None,
    **rope_kwargs,
) -> tuple[mx.array, float]:

    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta or config.rope_parameters["rope_theta"]
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (mx.arange(0, dim, 2, dtype=mx.int64).astype(mx.float32) / dim)
    )
    return inv_freq, attention_factor


class GLM4VRotaryEmbedding(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self._inv_freq = mx.array(inv_freq, dtype=mx.float32)
        self._original_inv_freq = mx.array(inv_freq, dtype=mx.float32)

    def __call__(self, x, position_ids):
        inv_freq_expanded = self._inv_freq[None, None, :, None].astype(mx.float32)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded, (3, position_ids.shape[1], self._inv_freq.shape[0], 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        freqs = (
            inv_freq_expanded.astype(mx.float32)
            @ position_ids_expanded.astype(mx.float32)
        ).transpose(0, 1, 3, 2)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section):
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.
    Args:
        q (mx.array): The query tensor.
        k (mx.array): The key tensor.
        cos (mx.array): The cosine part of the rotary embedding.
        sin (mx.array): The sine part of the rotary embedding.
        mrope_section (List[int]): Multimodal rope section for channel dimension of temporal, height and width.
        unsqueeze_dim (int, optional): Dimension to unsqueeze. Defaults to 1.
    Returns:
        tuple(mx.array): The rotated query and key tensors.
    """

    mrope_section = np.cumsum(mrope_section * 2)[:-1].tolist()

    cos = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(cos, mrope_section, axis=-1))], axis=-1
    )[
        :, None, :, :
    ]  # unsqueeze dim 1
    sin = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(sin, mrope_section, axis=-1))], axis=-1
    )[:, None, :, :]

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)

    return q_embed, k_embed


class Glm4vMoeAttention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope_parameters = args.rope_parameters or args.rope_scaling

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings

        queries, keys = apply_multimodal_rotary_pos_emb(
            queries, keys, cos, sin, self.rope_parameters["mrope_section"]
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, mx.array):
            mask = mask[..., : keys.shape[-2]]

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Glm4vMoeMLP(nn.Module):
    def __init__(
        self, config: TextConfig, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        down_proj = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        k = top_k
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4vMoeMLP(
                config=config, intermediate_size=intermediate_size
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class Glm4vMoeDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Glm4vMoeAttention(config)
        self.mlp = (
            MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
            )
            else Glm4vMoeMLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[mx.array] = None,
    ) -> mx.array:

        r = self.self_attn(self.input_layernorm(x), mask, cache, position_embeddings)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class GLM4VModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm4vMoeDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = GLM4VRotaryEmbedding(config)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:

        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds.astype(self.norm.weight.dtype)

        if position_ids is None:
            position_ids = mx.arange(cache[0].offset, cache[0].offset + h.shape[-2])
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

        position_embeddings = self.rotary_emb(h, position_ids)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * self.num_layers

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i], position_embeddings)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = GLM4VModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._rope_deltas = None

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        # Calculate RoPE index for image/video tokens
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = mx.where(
                    attention_mask[i] == 1, input_ids, mx.zeros_like(input_ids)
                )
                image_nums, video_nums = 0, 0
                vision_start_indices = mx.sum(
                    mx.where(
                        input_ids == vision_start_token_id,
                        mx.arange(input_ids.shape[0]),
                        mx.zeros_like(input_ids),
                    )
                )
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len))
                    index = index + st_idx
                    llm_pos_ids_list.append(index)
                    t_index = mx.arange(llm_grid_t).reshape(
                        llm_grid_t, 1
                    )  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(
                        t_index, (llm_grid_t, llm_grid_h * llm_grid_w)
                    )  # Equivalent to expand()
                    t_index = t_index.flatten()  # Flattens to 1D

                    h_index = mx.arange(llm_grid_h).reshape(
                        1, llm_grid_h, 1
                    )  # Equivalent to .view(1, -1)
                    h_index = mx.broadcast_to(
                        h_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )  # Equivalent to expand()
                    h_index = h_index.flatten()  # Flattens to 1D

                    w_index = mx.arange(llm_grid_w).reshape(
                        1, 1, llm_grid_w
                    )  # Equivalent to .view(1, -1)
                    w_index = mx.broadcast_to(
                        w_index, (llm_grid_t, llm_grid_h, llm_grid_w)
                    )  # Equivalent to expand()
                    w_index = w_index.flatten()  # Flattens to 1D

                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st

                    t_index = mx.arange(text_len).reshape(
                        1, text_len
                    )  # Equivalent to .view(-1, 1)
                    t_index = mx.broadcast_to(
                        t_index, (3, text_len)
                    )  # Equivalent to expand(3, -1)

                    llm_pos_ids_list.append(t_index + st_idx)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [
                        position_ids[:, :i, :],
                        new_positions,
                        position_ids[:, i + 1 :, :],
                    ],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = mx.array(mrope_position_deltas)[0]
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
                position_ids = mx.where(
                    attention_mask == 0, mx.ones_like(position_ids), position_ids
                )
                position_ids = mx.expand_dims(position_ids[0], axis=0)
                position_ids = mx.tile(position_ids, (3, 1, 1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(
                    -1, keepdims=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
                position_ids = mx.broadcast_to(
                    position_ids, (3, input_ids.shape[0], input_ids.shape[1])
                )
                mrope_position_deltas = mx.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

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
        # reset rope_deltas when processing a new image/video
        if pixel_values is not None:
            self._rope_deltas = None

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        if position_ids is None and (mask is None or mask.ndim == 2):
            # Calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache is not None and cache[0] is not None and (cache_offset == 0))
                or self._rope_deltas is None
                or cache is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    inputs, image_grid_thw, video_grid_thw, mask
                )
                self._rope_deltas = rope_deltas
            else:
                # Use the prev pre-calculated rope-deltas to get the correct position ids
                batch_size, seq_length = inputs.shape
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)

                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        # Slice delta to match batch
                        delta = delta[:batch_size]

                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        out = self.model(
            inputs, cache=cache, inputs_embeds=inputs_embeds, position_ids=position_ids
        )

        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        mpt_layer = self.args.num_hidden_layers

        # Stack experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        # Remove multi-token prediction layer
        return {
            k: v
            for k, v in weights.items()
            if not k.startswith(f"language_model.model.layers.{mpt_layer}")
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
