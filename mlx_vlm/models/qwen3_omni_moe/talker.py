from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

from mlx_vlm.models.qwen3_omni_moe.config import (
    CodePredictorConfig,
    TalkerConfig,
    TextConfig,
)
from mlx_vlm.sample_utils import top_p_sampling

from ..base import create_attention_mask, scaled_dot_product_attention
from ..cache import KVCache
from .language import Attention, Qwen3OmniMoeThinkerTextRotaryEmbedding


class CodePredictorRotaryEmbedding:
    def __init__(self, config: CodePredictorConfig):
        self.config = config
        head_dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
        )
        self.inv_freq = inv_freq
        self.attention_scaling = 1.0

    def __call__(
        self, x: mx.array, position_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        batch_size = position_ids.shape[0]
        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, :, None].astype(mx.float32),
            (batch_size, self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = mx.expand_dims(position_ids.astype(mx.float32), axis=1)
        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 1, 2)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half_code(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_code(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (rotate_half_code(q) * sin)
    k_embed = (k * cos) + (rotate_half_code(k) * sin)
    return q_embed, k_embed


class CodePredictorMLP(nn.Module):
    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act == "silu":
            self.act_fn = nn.silu
        elif config.hidden_act == "gelu":
            self.act_fn = nn.gelu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approx="precise")
        else:
            self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CodePredictorAttention(nn.Module):
    def __init__(self, config: CodePredictorConfig, idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(dims=self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if (
                hasattr(config, "layer_types")
                and config.layer_types
                and idx < len(config.layer_types)
                and config.layer_types[idx] == "sliding_attention"
            )
            else None
        )
        self.rotary_emb = CodePredictorRotaryEmbedding(config)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        B, L, D = hidden_states.shape
        hidden_shape = (B, L, -1, self.head_dim)

        query_states = (
            self.q_proj(hidden_states).reshape(*hidden_shape).transpose(0, 2, 1, 3)
        )
        key_states = (
            self.k_proj(hidden_states).reshape(*hidden_shape).transpose(0, 2, 1, 3)
        )
        value_states = (
            self.v_proj(hidden_states).reshape(*hidden_shape).transpose(0, 2, 1, 3)
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if position_embeddings is None:
            if position_ids is None:
                if past_key_values is not None:
                    offset = (
                        past_key_values.offset
                        if hasattr(past_key_values, "offset")
                        else 0
                    )
                    position_ids = mx.arange(offset, offset + L)
                else:
                    position_ids = mx.arange(L)
                position_ids = mx.expand_dims(position_ids, axis=0)
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb_code(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update_and_fetch(
                key_states, value_states
            )

        if attention_mask is not None and isinstance(attention_mask, mx.array):
            kv_seq_len = key_states.shape[-2]
            if attention_mask.shape[-1] != kv_seq_len:
                attention_mask = attention_mask[..., :kv_seq_len]

        if self.is_causal and attention_mask is None:
            attention_mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            attention_mask = attention_mask.astype(query_states.dtype)

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            past_key_values,
            scale=self.scaling,
            mask=attention_mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class CodePredictorDecoderLayer(nn.Module):
    def __init__(self, config: CodePredictorConfig, idx: int):
        super().__init__()
        self.self_attn = CodePredictorAttention(config, idx)
        self.mlp = CodePredictorMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = (
            config.layer_types[idx]
            if hasattr(config, "layer_types") and config.layer_types
            else "full_attention"
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[mx.array] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CodePredictorModel(nn.Module):
    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.config = config
        self.layers = [
            CodePredictorDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = CodePredictorRotaryEmbedding(config)
        self.codec_embedding = [
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ]

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mx.array] = None,
        generation_steps: Optional[int] = None,
    ) -> mx.array:
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")

        if use_cache and past_key_values is None:
            past_key_values = [KVCache() for _ in range(len(self.layers))]

        if cache_position is None:
            if past_key_values is not None and len(past_key_values) > 0:
                offset = (
                    past_key_values[0].offset
                    if hasattr(past_key_values[0], "offset")
                    else 0
                )
            else:
                offset = 0
            cache_position = mx.arange(offset, offset + inputs_embeds.shape[1])

        if position_ids is None:
            position_ids = mx.expand_dims(cache_position, axis=0)

        if attention_mask is None:
            attention_mask = create_attention_mask(
                inputs_embeds,
                past_key_values[0] if past_key_values else None,
            )

        if attention_mask is not None and not isinstance(attention_mask, dict):
            causal_mask_mapping = {
                "full_attention": attention_mask,
            }
        else:
            causal_mask_mapping = (
                attention_mask
                if isinstance(attention_mask, dict)
                else {"full_attention": None}
            )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping.get(
                    decoder_layer.attention_type,
                    causal_mask_mapping.get("full_attention"),
                ),
                position_ids=position_ids,
                past_key_values=past_key_values[i] if past_key_values else None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class CodePredictor(nn.Module):
    def __init__(self, config: CodePredictorConfig):
        super().__init__()
        self.config = config
        self.model = CodePredictorModel(config)
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ]

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[mx.array] = None,
        generation_steps: Optional[int] = None,
    ):
        if (
            inputs_embeds is not None
            and inputs_embeds.shape[1] > 1
            and generation_steps is None
        ):
            generation_steps = inputs_embeds.shape[1] - 2
        elif input_ids is not None and generation_steps is not None:
            inputs_embeds = self.model.codec_embedding[generation_steps - 1](input_ids)

        if generation_steps is None:
            generation_steps = 0

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            generation_steps=generation_steps,
        )

        hidden_states = outputs
        logits = self.lm_head[generation_steps](hidden_states)

        return logits, hidden_states, inputs_embeds


class TalkerResizeMlp(nn.Module):
    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.thinker_hidden_size, config.text_config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.silu(self.linear_fc1(x)))


class TalkerTextMlp(nn.Module):
    def __init__(self, config: TextConfig, intermediate_sz: int):
        super().__init__()
        if not intermediate_sz:
            intermediate_sz = config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_sz, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_sz, bias=False)
        self.down_proj = nn.Linear(intermediate_sz, config.hidden_size, bias=False)

        if config.hidden_act == "silu":
            self.act_fn = nn.silu
        elif config.hidden_act == "gelu":
            self.act_fn = nn.gelu
        else:
            self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TalkerSparseMoeBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.num_experts
        )
        self.shared_expert = TalkerTextMlp(
            config, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        router_logits = self.gate(hidden_states)
        routing_weights = mx.softmax(
            router_logits.astype(mx.float32), axis=-1, precise=True
        )

        k = self.top_k
        inds = mx.argpartition(routing_weights, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(routing_weights, inds, axis=-1)

        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)

        y = self.switch_mlp(hidden_states, inds)
        final_hidden_states = (y * scores[..., None]).sum(axis=-2)

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_gate_output = nn.sigmoid(self.shared_expert_gate(hidden_states))
        shared_expert_output = shared_expert_gate_output * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states, router_logits


class TalkerModelDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, idx: int):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = TalkerSparseMoeBlock(config)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if position_ids is not None and position_ids.ndim == 2:
            position_ids_3d = mx.tile(mx.expand_dims(position_ids, axis=0), (3, 1, 1))
        else:
            position_ids_3d = position_ids

        hidden_states = self.self_attn(
            hidden_states,
            mask=attention_mask,
            cache=past_key_values,
            position_ids=position_ids_3d,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TalkerModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.layers = [
            TalkerModelDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[mx.array] = None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[list] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.codec_embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = [KVCache() for _ in range(len(self.layers))]

        if cache_position is None:
            if past_key_values is not None and len(past_key_values) > 0:
                offset = (
                    past_key_values[0].offset
                    if hasattr(past_key_values[0], "offset")
                    else 0
                )
            else:
                offset = 0
            cache_position = mx.arange(offset, offset + inputs_embeds.shape[1])

        if position_ids is None:
            position_ids = cache_position
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        text_position_ids = (
            position_ids[0] if position_ids.shape[0] >= 1 else position_ids
        )

        if attention_mask is None:
            attention_mask = create_attention_mask(
                inputs_embeds,
                past_key_values if past_key_values else None,
            )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values[layer_idx] if past_key_values else None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if deepstack_visual_embeds is not None and layer_idx < len(
                deepstack_visual_embeds
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if layer_idx % 4 == 0:
                mx.eval(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _deepstack_process(
        self,
        hidden_states: mx.array,
        visual_pos_masks: mx.array,
        visual_embeds: mx.array,
    ):
        if visual_pos_masks.ndim == 3:
            visual_pos_masks = visual_pos_masks[..., 0]
        visual_embeds = visual_embeds.astype(hidden_states.dtype)
        visual_indices = np.where(visual_pos_masks)[0].tolist()
        local_this = hidden_states[:, visual_indices, :] + visual_embeds
        hidden_states[:, visual_indices, :] = local_this
        return hidden_states


class Talker(nn.Module):
    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config
        self.model = TalkerModel(config.text_config)
        self.text_projection = TalkerResizeMlp(config)
        self.hidden_projection = TalkerResizeMlp(config)
        self.code_predictor = CodePredictor(config.code_predictor_config)
        self.codec_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[mx.array] = None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[list] = None,
        generation_steps: Optional[int] = None,
        residual_codes: Optional[mx.array] = None,
        trailing_text_hidden: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.codec_embedding(input_ids)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        hidden_states = outputs
        logits = self.codec_head(hidden_states)

        return logits, hidden_states

    def prepare_inputs_for_generation(
        self,
        input_ids: mx.array,
        past_hidden: mx.array,
        trailing_text_hidden: mx.array,
        tts_pad_embed: mx.array,
        generation_step: int,
        temperature: float = 1.0,
        top_p: float = 0.8,
    ):
        token = input_ids
        last_id_hidden = self.model.codec_embedding(token)

        cp_input_embeds = mx.concatenate([past_hidden, last_id_hidden], axis=1)
        cp_past_key_values = [
            KVCache() for _ in range(len(self.code_predictor.model.layers))
        ]

        cp_logits, cp_hidden, _ = self.code_predictor(
            inputs_embeds=cp_input_embeds,
            past_key_values=cp_past_key_values,
            use_cache=True,
        )

        if temperature == 0:
            cp_token = mx.argmax(cp_logits[:, -1, :], axis=-1)
        else:
            cp_token = top_p_sampling(cp_logits[:, -1, :], top_p, temperature)

        current_step_codes = [token, cp_token[:, None]]

        mid_residual_hiddens = []

        for cp_step in range(1, self.config.num_code_groups - 1):
            cp_logits, cp_hidden, cp_input_embeds_out = self.code_predictor(
                input_ids=cp_token[:, None],
                past_key_values=cp_past_key_values,
                use_cache=True,
                generation_steps=cp_step,
            )
            mid_residual_hiddens.append(cp_input_embeds_out)

            if temperature == 0:
                cp_token = mx.argmax(cp_logits[:, -1, :], axis=-1)
            else:
                cp_token = top_p_sampling(cp_logits[:, -1, :], top_p, temperature)

            current_step_codes.append(cp_token[:, None])

        last_residual_hidden = self.code_predictor.model.codec_embedding[-1](
            cp_token[:, None]
        )

        codec_hiddens = [last_id_hidden] + mid_residual_hiddens + [last_residual_hidden]
        codec_hiddens_stacked = mx.concatenate(codec_hiddens, axis=1)
        inputs_embeds = mx.sum(codec_hiddens_stacked, axis=1, keepdims=True)

        if generation_step < trailing_text_hidden.shape[1]:
            trailing = trailing_text_hidden[:, generation_step].reshape(1, 1, -1)
            inputs_embeds = inputs_embeds + trailing
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        residual_codes = mx.concatenate(current_step_codes, axis=1)

        return inputs_embeds, residual_codes

    def generate(
        self,
        inputs_embeds: mx.array,
        trailing_text_hidden: mx.array,
        tts_pad_embed: mx.array,
        talker_input_ids: mx.array,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_p: float = 1.0,
        **kwargs,
    ):
        past_key_values = [
            KVCache() for _ in range(self.config.text_config.num_hidden_layers)
        ]

        logits, hidden_states = self(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )

        hidden_states_list = [(hidden_states, None)]

        if temperature == 0:
            token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            token = top_p_sampling(logits[:, -1, :], top_p, temperature)

        generation_step = 0

        for _ in range(max_new_tokens):
            token_scalar = token.item()
            if token_scalar == self.config.codec_eos_token_id:
                break

            past_hidden = hidden_states_list[-1][0][:, -1:]
            inputs_embeds, residual_codes = self.prepare_inputs_for_generation(
                input_ids=token[:, None],
                past_hidden=past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
                temperature=temperature,
                top_p=0.8,
            )

            logits, hidden_states = self(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )

            hidden_states_list.append((hidden_states, residual_codes))

            if temperature == 0:
                token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                token = top_p_sampling(logits[:, -1, :], top_p, temperature)

            generation_step += 1

        class TalkerGenerateResult:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        return TalkerGenerateResult(hidden_states_list)

    def generate_stream(
        self,
        inputs_embeds: mx.array,
        trailing_text_hidden: mx.array,
        tts_pad_embed: mx.array,
        talker_input_ids: mx.array,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_p: float = 1.0,
        **kwargs,
    ):
        past_key_values = [
            KVCache() for _ in range(self.config.text_config.num_hidden_layers)
        ]
        logits, hidden_states = self(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )

        if temperature == 0:
            token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            token = top_p_sampling(logits[:, -1, :], top_p, temperature)

        generation_step = 0
        past_hidden = hidden_states[:, -1:]

        for _ in range(max_new_tokens):
            token_scalar = token.item()
            if token_scalar == self.config.codec_eos_token_id:
                break

            inputs_embeds, residual_codes = self.prepare_inputs_for_generation(
                input_ids=token[:, None],
                past_hidden=past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
                temperature=temperature,
                top_p=0.8,
            )

            logits, hidden_states = self(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_hidden = hidden_states[:, -1:]

            yield residual_codes

            if temperature == 0:
                token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                token = top_p_sampling(logits[:, -1, :], top_p, temperature)

            generation_step += 1

    def sanitize(self, weights):
        for l in range(self.config.text_config.num_hidden_layers):
            prefix = f"talker.model.layers.{l}.mlp"
            for n in ["gate_proj", "down_proj", "up_proj"]:
                experts_weights = []
                for e in range(self.config.text_config.num_experts):
                    key = f"{prefix}.experts.{e}.{n}.weight"
                    if key in weights:
                        experts_weights.append(weights.pop(key))

                if experts_weights:
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(
                        experts_weights, axis=0
                    )
        return weights
