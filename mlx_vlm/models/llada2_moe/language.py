import sys
import time
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import KVCache
from .config import ModelConfig


def _topk(x: mx.array, k: int, axis: int = -1) -> Tuple[mx.array, mx.array]:
    indices = mx.argpartition(-x, kth=k - 1, axis=axis)[..., :k]
    values = mx.take_along_axis(x, indices, axis=axis)
    order = mx.argsort(-values, axis=axis)
    return mx.take_along_axis(values, order, axis=axis), mx.take_along_axis(
        indices, order, axis=axis
    )


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    raise ValueError(f"Unsupported LLaDA2 MoE score function: {func}")


def _make_bidirectional_mask(
    attention_mask: Optional[mx.array], x: mx.array
) -> Optional[mx.array]:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        if attention_mask.dtype == mx.bool_:
            return attention_mask
        return mx.where(attention_mask.astype(mx.bool_), 0.0, mx.finfo(x.dtype).min)
    if attention_mask.ndim != 2:
        return attention_mask

    if attention_mask.shape[-1] == 0:
        return None
    if bool(mx.all(attention_mask).item()):
        return None
    mask = attention_mask[:, None, None, :].astype(mx.bool_)
    return mx.where(mask, 0.0, mx.finfo(x.dtype).min).astype(x.dtype)


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.score_function = config.score_function
        self.weight = mx.zeros((config.num_experts, config.hidden_size))
        self.expert_bias = mx.zeros((config.num_experts,))

    def _group_limited_topk(self, scores: mx.array):
        group_size = self.num_experts // self.n_group
        group_scores = scores.reshape(*scores.shape[:-1], self.n_group, group_size)
        group_scores = _topk(group_scores, k=2, axis=-1)[0].sum(axis=-1)
        group_idx = _topk(group_scores, k=self.topk_group, axis=-1)[1]

        expert_groups = mx.arange(self.num_experts, dtype=group_idx.dtype) // group_size
        group_mask = (expert_groups[None, :] == group_idx[..., None]).any(axis=-2)
        masked_scores = mx.where(group_mask, scores, mx.finfo(scores.dtype).min)
        return _topk(masked_scores, k=self.top_k, axis=-1)

    def __call__(self, hidden_states: mx.array):
        logits = hidden_states.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        scores = _score_func(logits, self.score_function)
        _, topk_idx = self._group_limited_topk(scores + self.expert_bias)
        topk_weight = mx.take_along_axis(scores, topk_idx, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(axis=-1, keepdims=True) + 1e-20
            )
        topk_weight = topk_weight * self.routed_scaling_factor
        return (
            topk_idx.astype(mx.int32),
            topk_weight.astype(hidden_states.dtype),
            logits,
        )


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = LLaDA2MoeGate(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.num_experts
        )
        self.shared_experts = None
        if config.num_shared_experts:
            self.shared_experts = LLaDA2MoeMLP(
                config, config.moe_intermediate_size * config.num_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores, _ = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class LLaDA2MoeAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rotary_dim
        self.scale = self.head_dim**-0.5
        self.query_key_value = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        if config.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.query_layernorm = None
            self.key_layernorm = None
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias
        )
        self.rope_theta = config.rope_theta

    def _apply_rope(self, q: mx.array, k: mx.array, offset: Any = 0):
        if self.rope_dim <= 0:
            return q, k
        q_rot = mx.fast.rope(
            q[..., : self.rope_dim],
            self.rope_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )
        k_rot = mx.fast.rope(
            k[..., : self.rope_dim],
            self.rope_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )
        q = mx.concatenate([q_rot, q[..., self.rope_dim :]], axis=-1)
        k = mx.concatenate([k_rot, k[..., self.rope_dim :]], axis=-1)
        return q, k

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        qkv = self.query_key_value(x).reshape(
            B, L, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
        )
        queries, keys, values = mx.split(
            qkv, [self.num_heads, self.num_heads + self.num_key_value_heads], axis=-2
        )

        if self.query_layernorm is not None:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None and position_ids is None else 0
        queries, keys = self._apply_rope(queries, keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attention = LLaDA2MoeAttention(config, layer_idx)
        if config.num_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.mlp = LLaDA2MoeSparseMoeBlock(config)
        else:
            self.mlp = LLaDA2MoeMLP(config, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.attention(
            self.input_layernorm(x), mask=mask, cache=cache, position_ids=position_ids
        )
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class LLaDA2MoeModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            LLaDA2MoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ):
        h = self.word_embeddings(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        mask = _make_bidirectional_mask(mask if mask is not None else attention_mask, h)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=layer_cache, position_ids=position_ids)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = LLaDA2MoeModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            attention_mask=kwargs.get("attention_mask"),
            cache=cache,
            position_ids=kwargs.get("position_ids"),
        )
        if self.config.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    @staticmethod
    def _top_k_logits(logits: mx.array, k: Optional[int]) -> mx.array:
        if k is None or k <= 0:
            return logits
        values = _topk(logits, k=k, axis=-1)[0]
        return mx.where(logits < values[..., -1:], mx.finfo(logits.dtype).min, logits)

    @staticmethod
    def _top_p_logits(logits: mx.array, p: Optional[float]) -> mx.array:
        if p is None or p >= 1.0:
            return logits
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = mx.cumsum(
            mx.softmax(sorted_logits, axis=-1, precise=True), axis=-1
        )
        sorted_mask = cumulative_probs > p
        sorted_mask = mx.concatenate(
            [mx.zeros_like(sorted_mask[..., :1]), sorted_mask[..., :-1]], axis=-1
        )
        inverse_indices = mx.argsort(sorted_indices, axis=-1)
        mask = mx.take_along_axis(sorted_mask, inverse_indices, axis=-1)
        return mx.where(mask, mx.finfo(logits.dtype).min, logits)

    def _sample_with_temperature_topk_topp(
        self,
        logits: mx.array,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        if temperature == 0.0:
            token = mx.argmax(logits, axis=-1)
            probs = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
            token_prob = mx.take_along_axis(probs, token[..., None], axis=-1)[..., 0]
            return token, token_prob

        if temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        token = mx.random.categorical(logits.astype(mx.float32), axis=-1)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
        token_prob = mx.take_along_axis(probs, token[..., None], axis=-1)[..., 0]
        return token, token_prob

    def generate(
        self,
        inputs: mx.array,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        editing_threshold: float = 0.9,
        max_post_steps: int = 16,
        eos_id: Optional[int] = None,
        mask_id: Optional[int] = None,
        num_to_transfer: int = 1,
        visualize: bool = False,
        tokenizer: Optional[Any] = None,
        skip_special_tokens: bool = False,
        stats: Optional[Dict[str, float]] = None,
    ) -> mx.array:
        if inputs.shape[0] != 1:
            raise ValueError(
                "LLaDA2 MoE diffusion generation currently supports batch size 1."
            )

        eos_id = self.config.eos_token_id if eos_id is None else eos_id
        mask_id = self.config.mask_token_id if mask_id is None else mask_id
        eos_token_ids = (
            set(eos_id) if isinstance(eos_id, (list, tuple, set)) else {eos_id}
        )
        steps = max(1, min(steps, max(1, gen_length // minimal_topk)))
        visualizer_state = {"active": visualize and sys.stdout.isatty()}

        def clear_visualizer() -> None:
            if not visualizer_state["active"]:
                return
            print("\r\033[2K\033[?7h", end="", flush=True)

        def decode_token(token_id: int) -> str:
            if tokenizer is None:
                return str(token_id)
            piece = tokenizer.decode(
                [token_id], skip_special_tokens=skip_special_tokens
            )
            return piece.replace("\n", "\\n") or " "

        def visualize_tokens(tokens: mx.array) -> None:
            if not visualizer_state["active"]:
                return
            pieces = []
            token_ids = tokens[0].tolist()
            for i, token_id in enumerate(token_ids):
                if token_id == mask_id:
                    pieces.append("[MASK]")
                elif token_id in eos_token_ids:
                    pieces.append(decode_token(token_id) or "<eos>")
                    tail = len(token_ids) - i - 1
                    if tail:
                        pieces.extend(["[MASK]"] * tail)
                    break
                else:
                    pieces.append(decode_token(token_id))

            line = " ".join(pieces)
            clear_visualizer()
            print("\033[?7l" + line + "\033[?7h", end="", flush=True)

        prompt_length = inputs.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = mx.tril(mx.ones((num_blocks, num_blocks), dtype=mx.bool_))
        block_attention_mask = mx.repeat(
            mx.repeat(block_mask, block_length, axis=0), block_length, axis=1
        )[None, None]

        x = mx.full((1, total_length), mask_id, dtype=inputs.dtype)
        x = mx.concatenate([inputs, x[:, prompt_length:]], axis=1)
        position_ids = mx.arange(total_length, dtype=mx.int32)[None, :]
        prefill_blocks = prompt_length // block_length
        display_end = prompt_length + gen_length
        visualize_tokens(x[:, prompt_length:display_end])
        prompt_tic = time.perf_counter()
        recorded_prompt_time = False

        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            block_start = num_block * block_length
            cur_x = x[:, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            cur_attn_mask = block_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            block_positions = mx.arange(block_length)
            prompt_mask = block_start + block_positions < prompt_length

            post_steps = 0
            while True:
                old_block = cur_x[:, -block_length:]
                active_block_mask = old_block == mask_id
                if not bool(active_block_mask.any().item()):
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                logits = self(
                    cur_x,
                    mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                ).logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                if stats is not None and not recorded_prompt_time:
                    mx.eval(x0, x0_p)
                    stats["prompt_time"] = time.perf_counter() - prompt_tic
                    stats["prompt_tokens"] = float(prompt_length)
                    recorded_prompt_time = True

                transfer_mask = mx.zeros_like(active_block_mask)
                if bool(active_block_mask.any().item()):
                    confidence = mx.where(active_block_mask, x0_p, -mx.inf)
                    high_confidence = (confidence > threshold) & active_block_mask
                    if int(high_confidence.sum().item()) >= num_to_transfer:
                        transfer_mask = high_confidence
                    else:
                        available = int(active_block_mask.sum().item())
                        _, indices = _topk(confidence, min(num_to_transfer, available))
                        transfer_mask = (
                            block_positions[None, None, :] == indices[..., None]
                        ).any(axis=1)

                editable = (~active_block_mask) & (~prompt_mask[None, :])
                edit_confidence = mx.where(editable, x0_p, -mx.inf)
                edit_mask = (
                    (edit_confidence > editing_threshold) & editable & (x0 != old_block)
                )
                final_mask = transfer_mask | edit_mask

                new_block = mx.where(final_mask, x0, old_block)
                cur_x = mx.concatenate([cur_x[:, :-block_length], new_block], axis=1)
                x = mx.concatenate([cur_x, x[:, current_window_end:]], axis=1)
                if bool(final_mask.any().item()):
                    visualize_tokens(x[:, prompt_length:display_end])

                if not bool(active_block_mask.any().item()) and not bool(
                    edit_mask.any().item()
                ):
                    break

            x = mx.concatenate([cur_x, x[:, current_window_end:]], axis=1)
            if eos_early_stop:
                generated = x[0, prompt_length:current_window_end]
                if not bool((generated == mask_id).any().item()):
                    if bool((generated == eos_id).any().item()):
                        break

        generated = x[:, prompt_length : prompt_length + gen_length]
        generated_ids = generated[0].tolist()
        end = generated_ids.index(eos_id) + 1 if eos_id in generated_ids else gen_length
        clear_visualizer()
        return generated[:, :end]

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        for layer_idx in range(self.config.num_hidden_layers):
            if layer_idx < self.config.first_k_dense_replace:
                continue
            prefix = f"language_model.model.layers.{layer_idx}.mlp"
            for module in ("gate_proj", "down_proj", "up_proj"):
                for suffix in ("weight", "scales", "biases"):
                    first_key = f"{prefix}.experts.0.{module}.{suffix}"
                    if first_key not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{module}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{module}.{suffix}")
                            for e in range(self.config.num_experts)
                        ]
                    )

        if self.config.tie_word_embeddings:
            weights.pop("language_model.lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
