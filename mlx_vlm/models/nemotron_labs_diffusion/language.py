import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.rope_utils import initialize_rope

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig


def _topk(x: mx.array, k: int, axis: int = -1) -> Tuple[mx.array, mx.array]:
    indices = mx.argpartition(-x, kth=k - 1, axis=axis)[..., :k]
    values = mx.take_along_axis(x, indices, axis=axis)
    order = mx.argsort(-values, axis=axis)
    return mx.take_along_axis(values, order, axis=axis), mx.take_along_axis(
        indices, order, axis=axis
    )


def _first_token_index(tokens: mx.array, token_ids: set[int]) -> Optional[int]:
    values = tokens.tolist()
    return next(
        (index for index, token_id in enumerate(values) if token_id in token_ids),
        None,
    )


def _wrap_text(text: str, width: int) -> str:
    lines = []
    while len(text) > width:
        split_at = text.rfind(" ", 0, width + 1)
        if split_at <= 0:
            split_at = width
        lines.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    if text:
        lines.append(text)
    return "\n".join(lines)


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

    if attention_mask.shape[-1] == 0 or bool(mx.all(attention_mask).item()):
        return None
    mask = attention_mask[:, None, None, :].astype(mx.bool_)
    return mx.where(mask, 0.0, mx.finfo(x.dtype).min).astype(x.dtype)


def _llama4_attention_scale(
    config: ModelConfig, length: int, offset: Any, dtype: mx.Dtype
) -> mx.array:
    beta = config.rope_parameters.get("llama_4_scaling_beta")
    original_max = config.rope_parameters.get("original_max_position_embeddings")
    if beta is None or original_max is None:
        return mx.array(1.0, dtype=dtype)
    positions = mx.arange(length, dtype=mx.float32) + offset
    scale = 1.0 + float(beta) * mx.log1p(mx.floor(positions / float(original_max)))
    return scale.astype(dtype)[None, None, :, None]


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DraftLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.linear = linear
        self.scale = scale
        out_dim, in_dim = linear.weight.shape
        self.lora_a = mx.zeros((in_dim, rank), dtype=linear.weight.dtype)
        self.lora_b = mx.zeros((rank, out_dim), dtype=linear.weight.dtype)
        self.enabled = False

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if not self.enabled:
            return y
        z = (x @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)
        return y + (self.scale * z).astype(y.dtype)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        use_cache: bool = True,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, self.head_dim)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        queries = queries * _llama4_attention_scale(
            self.config, L, offset, queries.dtype
        )

        if cache is not None:
            if use_cache:
                keys, values = cache.update_and_fetch(keys, values)
            elif cache.keys is not None:
                keys = mx.concatenate(
                    [cache.keys[..., : cache.offset, :], keys], axis=2
                )
                values = mx.concatenate(
                    [cache.values[..., : cache.offset, :], values], axis=2
                )

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        use_cache: bool = True,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x), mask=mask, cache=cache, use_cache=use_cache
        )
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class NemotronLabsDiffusionEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        use_cache: bool = True,
        use_causal_mask: bool = False,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if use_causal_mask:
            layer_mask = create_attention_mask(h, cache[0])
        else:
            layer_mask = _make_bidirectional_mask(
                mask if mask is not None else attention_mask, h
            )
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask=layer_mask, cache=layer_cache, use_cache=use_cache)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dlm_paradigm not in ("bidirectional", "autoregressive"):
            raise ValueError(
                f"Unsupported Nemotron Labs Diffusion paradigm: {config.dlm_paradigm}"
            )
        self.config = config
        self.model_type = config.model_type
        self.model = NemotronLabsDiffusionEncoder(config)
        if not config.tie_word_embeddings:
            self.diffusion_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        self._linear_spec_lora_loaded = False

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
            use_cache=kwargs.get("use_cache", True),
            use_causal_mask=kwargs.get("use_causal_mask", True),
        )
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.diffusion_head(out)
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

    def _project_hidden(self, hidden_states: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        return self.diffusion_head(hidden_states)

    def _sample_tokens(
        self,
        logits: mx.array,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> mx.array:
        if temperature == 0.0:
            return mx.argmax(logits, axis=-1)

        if temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        return mx.random.categorical(logits.astype(mx.float32), axis=-1)

    @staticmethod
    def _trim_cache(cache, max_length: int) -> None:
        for layer_cache in cache:
            excess = max(0, int(layer_cache.offset) - int(max_length))
            if excess:
                layer_cache.trim(excess)

    def load_linear_spec_lora(self, adapter_path: str | Path) -> bool:
        adapter_path = Path(adapter_path)
        adapter_file = adapter_path / "adapter_model.safetensors"
        if not adapter_file.exists():
            return False
        weights = mx.load(str(adapter_file))
        rank = 128
        scale = 4.0

        for layer_idx, layer in enumerate(self.model.layers):
            o_proj = layer.self_attn.o_proj
            if not isinstance(o_proj, DraftLoRALinear):
                o_proj = DraftLoRALinear(o_proj, rank=rank, scale=scale)
                layer.self_attn.o_proj = o_proj

            prefix = "base_model.model.encoder.layers." f"{layer_idx}.self_attn.o_proj"
            key_a = f"{prefix}.lora_A.weight"
            key_b = f"{prefix}.lora_B.weight"
            if key_a not in weights or key_b not in weights:
                return False
            o_proj.lora_a = weights[key_a].T.astype(o_proj.linear.weight.dtype)
            o_proj.lora_b = weights[key_b].T.astype(o_proj.linear.weight.dtype)

        self._linear_spec_lora_loaded = True
        return True

    def set_linear_spec_lora_enabled(self, enabled: bool) -> None:
        for layer in self.model.layers:
            o_proj = layer.self_attn.o_proj
            if isinstance(o_proj, DraftLoRALinear):
                o_proj.enabled = enabled

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
        min_threshold: Optional[float] = None,
        editing_threshold: float = 0.9,
        max_post_steps: int = 16,
        eos_id: Optional[int] = None,
        mask_id: Optional[int] = None,
        num_to_transfer: int = 1,
        max_transfer_per_step: Optional[int] = None,
        stability_steps: int = 2,
        visualize: bool = False,
        tokenizer: Optional[Any] = None,
        skip_special_tokens: bool = False,
        stats: Optional[Dict[str, float]] = None,
        linear_speculative: bool = False,
    ) -> mx.array:
        if inputs.shape[0] != 1:
            raise ValueError(
                "Nemotron Labs Diffusion generation currently supports batch size 1."
            )

        eos_id = self.config.eos_token_id if eos_id is None else eos_id
        mask_id = self.config.mask_token_id if mask_id is None else mask_id
        if linear_speculative:
            if not self._linear_spec_lora_loaded:
                model_path = getattr(self, "model_path", None)
                if model_path is not None:
                    self.load_linear_spec_lora(Path(model_path) / "linear_spec_lora")
            output, _ = self.linear_spec_generate(
                inputs,
                max_new_tokens=gen_length,
                block_length=block_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                mask_token_id=mask_id,
                eos_token_id=eos_id,
                threshold=0.0,
                stats=stats,
            )
            return output[:, inputs.shape[1] :]

        eos_token_ids = (
            set(eos_id) if isinstance(eos_id, (list, tuple, set)) else {eos_id}
        )
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")
        steps = max(1, int(steps))
        if max_transfer_per_step is not None:
            max_transfer_per_step = min(
                block_length, max(1, int(max_transfer_per_step))
            )

        visualizer_state = {
            "active": visualize and sys.stdout.isatty(),
            "alternate_screen": False,
            "rows": 0,
            "last_draw": 0.0,
            "min_interval": 0.1,
            "token_ids": None,
            "pieces": None,
            "canvas": "",
        }

        def clear_visualizer() -> None:
            if not visualizer_state["active"]:
                return
            if visualizer_state["alternate_screen"]:
                print("\033[H\033[2J", end="", flush=True)
                visualizer_state["rows"] = 0
                return
            if visualizer_state["rows"] == 0:
                return
            controls = ["\r\033[2K"]
            for _ in range(visualizer_state["rows"] - 1):
                controls.append("\033[1A\r\033[2K")
            print("".join(controls), end="", flush=True)
            visualizer_state["rows"] = 0

        def finish_visualizer() -> None:
            if not visualizer_state["active"]:
                return
            if visualizer_state["alternate_screen"]:
                print("\033[H\033[2J\033[?25h\033[?1049l", end="", flush=True)
                visualizer_state["alternate_screen"] = False
                visualizer_state["rows"] = 0
            else:
                clear_visualizer()

        def decode_token(token_id: int) -> str:
            if tokenizer is None:
                return str(token_id)
            piece = tokenizer.decode(
                [token_id], skip_special_tokens=skip_special_tokens
            )
            return piece.replace("\n", "\\n") or " "

        def visualize_tokens(tokens: mx.array, force: bool = False) -> None:
            if not visualizer_state["active"]:
                return
            now = time.perf_counter()
            if (
                not force
                and now - visualizer_state["last_draw"]
                < visualizer_state["min_interval"]
            ):
                return
            token_ids = tokens[0].tolist()
            pieces = visualizer_state["pieces"]
            previous_token_ids = visualizer_state["token_ids"]
            if (
                pieces is None
                or previous_token_ids is None
                or len(previous_token_ids) != len(token_ids)
            ):
                pieces = ["[MASK]"] * len(token_ids)
                previous_token_ids = [mask_id] * len(token_ids)

            found_eos = False
            for i, token_id in enumerate(token_ids):
                previous_token_id = previous_token_ids[i]
                if found_eos:
                    if previous_token_id != mask_id:
                        pieces[i] = "[MASK]"
                    continue
                if token_id == mask_id:
                    if previous_token_id != mask_id:
                        pieces[i] = "[MASK]"
                elif token_id in eos_token_ids:
                    if previous_token_id != token_id:
                        pieces[i] = decode_token(token_id) or "<eos>"
                    found_eos = True
                elif previous_token_id != token_id:
                    pieces[i] = decode_token(token_id)

            visualizer_state["pieces"] = pieces
            visualizer_state["token_ids"] = token_ids
            terminal_size = shutil.get_terminal_size((120, 20))
            terminal_width = max(20, terminal_size.columns - 1)
            canvas = _wrap_text("".join(pieces), terminal_width)
            if not force and canvas == visualizer_state["canvas"]:
                return
            rows = max(1, canvas.count("\n") + 1)
            if (
                rows >= max(1, terminal_size.lines - 2)
                and not visualizer_state["alternate_screen"]
            ):
                print("\033[?1049h\033[?25l\033[H\033[2J", end="", flush=True)
                visualizer_state["alternate_screen"] = True
            clear_visualizer()
            print(canvas, end="", flush=True)
            visualizer_state["rows"] = rows
            visualizer_state["last_draw"] = now
            visualizer_state["canvas"] = canvas

        generated_blocks = []
        prompt_tic = time.perf_counter()
        recorded_prompt_time = False
        cache = self.make_cache()
        prefill_logits = self(
            inputs,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        ).logits
        next_token = self._sample_tokens(
            prefill_logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )[:, None]
        mx.eval(next_token)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(inputs.size)
            recorded_prompt_time = True

        total_generated = 0
        num_blocks = (gen_length + block_length - 1) // block_length
        for _ in range(num_blocks):
            remaining = gen_length - total_generated
            if remaining <= 0:
                break
            current_block_length = min(block_length, remaining)
            block_positions = mx.arange(block_length)
            block = mx.full((1, block_length), mask_id, dtype=inputs.dtype)
            block[:, 0] = next_token[:, 0]
            if visualizer_state["active"]:
                preview = (
                    mx.concatenate(generated_blocks + [block], axis=1)
                    if generated_blocks
                    else block
                )
                visualize_tokens(preview, force=True)

            for step_idx in range(steps):
                mask_index = block == mask_id
                if not bool(mask_index.any().item()):
                    break
                logits = self(
                    block,
                    cache=cache,
                    use_cache=False,
                    use_causal_mask=False,
                ).logits
                x0, token_probs = self._sample_with_temperature_topk_topp(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                if stats is not None and not recorded_prompt_time:
                    mx.eval(x0, token_probs)
                    stats["prompt_time"] = time.perf_counter() - prompt_tic
                    stats["prompt_tokens"] = float(inputs.size)
                    recorded_prompt_time = True
                x0 = mx.where(mask_index, x0, block)
                confidence = mx.where(mask_index, token_probs, -mx.inf)
                remaining_steps = max(1, steps - step_idx)
                masked_count = int(mask_index.sum().item())
                if threshold is not None:
                    transfer_count = masked_count
                else:
                    transfer_count = max(
                        1, (masked_count + remaining_steps - 1) // remaining_steps
                    )
                if max_transfer_per_step is not None:
                    transfer_count = min(transfer_count, max_transfer_per_step)
                _, indices = _topk(confidence, min(transfer_count, masked_count))
                transfer_mask = (
                    block_positions[None, None, :] == indices[..., None]
                ).any(axis=1)
                if threshold is not None:
                    high_confidence = (confidence >= threshold) & mask_index
                    if bool(high_confidence.any().item()):
                        transfer_mask = transfer_mask & high_confidence
                    else:
                        _, best_index = _topk(confidence, 1)
                        transfer_mask = (
                            block_positions[None, None, :] == best_index[..., None]
                        ).any(axis=1)
                block = mx.where(transfer_mask, x0, block)
                if visualizer_state["active"] and bool(transfer_mask.any().item()):
                    preview = (
                        mx.concatenate(generated_blocks + [block], axis=1)
                        if generated_blocks
                        else block
                    )
                    visualize_tokens(preview)

            output = self(
                block,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            )
            next_token = self._sample_tokens(
                output.logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[:, None]
            generated_block = block[:, :current_block_length]
            generated_blocks.append(generated_block)
            total_generated += current_block_length
            if (
                eos_early_stop
                and _first_token_index(generated_block[0], eos_token_ids) is not None
            ):
                break

        generated = (
            mx.concatenate(generated_blocks, axis=1)
            if generated_blocks
            else mx.zeros((1, 0), dtype=inputs.dtype)
        )
        generated_ids = generated[0].tolist()
        end = next(
            (
                i + 1
                for i, token_id in enumerate(generated_ids)
                if token_id in eos_token_ids
            ),
            generated.shape[1],
        )
        if visualizer_state["active"]:
            finish_visualizer()
            if tokenizer is not None:
                final_text = tokenizer.decode(
                    generated_ids[:end], skip_special_tokens=skip_special_tokens
                )
            else:
                final_text = " ".join(str(token_id) for token_id in generated_ids[:end])
            print(final_text, end="", flush=True)
            if stats is not None:
                stats["text_already_printed"] = True
        return generated[:, :end]

    def ar_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        stats: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> tuple[mx.array, int]:
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )

        prompt_tic = time.perf_counter()
        cache = self.make_cache()
        prefill = self(
            prompt_ids,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        ).logits
        mx.eval(prefill)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(prompt_ids.size)

        generated = []
        next_logits = prefill[:, -1, :]
        nfe = 0
        for _ in range(max_new_tokens):
            nfe += 1
            next_token = self._sample_tokens(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[:, None]
            generated.append(next_token)
            if bool(
                mx.array(
                    [token in eos_token_ids for token in next_token[:, 0].tolist()]
                )
                .all()
                .item()
            ):
                break
            next_logits = self(
                next_token,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            ).logits[:, -1, :]

        if not generated:
            return prompt_ids, nfe
        return (
            mx.concatenate([prompt_ids, mx.concatenate(generated, axis=1)], axis=1),
            nfe,
        )

    def linear_spec_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        threshold: float = 0.0,
        stats: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> tuple[mx.array, int]:
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch size 1.")
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")

        mask_token_id = (
            self.config.mask_token_id if mask_token_id is None else mask_token_id
        )
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )

        prompt_tic = time.perf_counter()
        cache = self.make_cache()
        prefill = self(
            prompt_ids,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        ).logits
        mx.eval(prefill)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(prompt_ids.size)

        next_token = self._sample_tokens(
            prefill[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )[:, None]
        generated = [next_token]
        total_generated = 1
        nfe = 1

        if next_token.item() in eos_token_ids:
            return mx.concatenate([prompt_ids, next_token], axis=1), nfe

        while total_generated < max_new_tokens:
            cache_len = cache[0].offset
            block = mx.full((1, block_length), mask_token_id, dtype=prompt_ids.dtype)
            block[:, 0] = next_token[:, 0]

            while bool((block == mask_token_id).any().item()):
                self.set_linear_spec_lora_enabled(True)
                draft_logits = self(
                    block,
                    cache=cache,
                    use_cache=False,
                    use_causal_mask=False,
                ).logits
                nfe += 1
                is_mask = block == mask_token_id
                if threshold > 0:
                    draft_tokens, draft_probs = self._sample_with_temperature_topk_topp(
                        draft_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    draft_conf = mx.where(is_mask, draft_probs, -mx.inf)
                    unmask = draft_conf >= threshold
                    if not bool(unmask.any().item()):
                        _, best_idx = _topk(draft_conf, 1)
                        positions = mx.arange(block_length)
                        unmask = (positions[None, None, :] == best_idx[..., None]).any(
                            axis=1
                        )
                    block = mx.where(unmask, draft_tokens, block)
                else:
                    draft_tokens = self._sample_tokens(
                        draft_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    block = mx.where(is_mask, draft_tokens, block)
                    break

            self.set_linear_spec_lora_enabled(False)
            verify_logits = self(
                block,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            ).logits
            nfe += 1
            ar_tokens = self._sample_tokens(
                verify_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            ar_token_ids = ar_tokens[0].tolist()
            block_ids = block[0].tolist()
            accepted = 1
            for i in range(block_length - 1):
                if ar_token_ids[i] == block_ids[i + 1]:
                    accepted += 1
                else:
                    break
            accepted = min(accepted, max_new_tokens - total_generated)
            accepted_tokens = ar_tokens[:, :accepted]
            generated.append(accepted_tokens)
            total_generated += accepted

            self._trim_cache(cache, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            eos_index = _first_token_index(accepted_tokens[0], eos_token_ids)
            if eos_index is not None:
                generated[-1] = accepted_tokens[:, : eos_index + 1]
                break

        return (
            mx.concatenate([prompt_ids, mx.concatenate(generated, axis=1)], axis=1),
            nfe,
        )

    def stream_linear_spec_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        threshold: float = 0.0,
        stats: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch size 1.")
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")

        mask_token_id = (
            self.config.mask_token_id if mask_token_id is None else mask_token_id
        )
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_token_ids = (
            set(eos_token_id)
            if isinstance(eos_token_id, (list, tuple, set))
            else {eos_token_id}
        )

        prompt_tic = time.perf_counter()
        cache = self.make_cache()
        prefill = self(
            prompt_ids,
            cache=cache,
            use_cache=True,
            use_causal_mask=True,
        ).logits
        mx.eval(prefill)
        if stats is not None:
            stats["prompt_time"] = time.perf_counter() - prompt_tic
            stats["prompt_tokens"] = float(prompt_ids.size)

        next_token = self._sample_tokens(
            prefill[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )[:, None]
        mx.eval(next_token)
        yield next_token
        total_generated = 1

        if next_token.item() in eos_token_ids:
            return

        while total_generated < max_new_tokens:
            cache_len = cache[0].offset
            block = mx.full((1, block_length), mask_token_id, dtype=prompt_ids.dtype)
            block[:, 0] = next_token[:, 0]

            while bool((block == mask_token_id).any().item()):
                self.set_linear_spec_lora_enabled(True)
                draft_logits = self(
                    block,
                    cache=cache,
                    use_cache=False,
                    use_causal_mask=False,
                ).logits
                is_mask = block == mask_token_id
                if threshold > 0:
                    draft_tokens, draft_probs = self._sample_with_temperature_topk_topp(
                        draft_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    draft_conf = mx.where(is_mask, draft_probs, -mx.inf)
                    unmask = draft_conf >= threshold
                    if not bool(unmask.any().item()):
                        _, best_idx = _topk(draft_conf, 1)
                        positions = mx.arange(block_length)
                        unmask = (positions[None, None, :] == best_idx[..., None]).any(
                            axis=1
                        )
                    block = mx.where(unmask, draft_tokens, block)
                else:
                    draft_tokens = self._sample_tokens(
                        draft_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    block = mx.where(is_mask, draft_tokens, block)
                    break

            self.set_linear_spec_lora_enabled(False)
            verify_logits = self(
                block,
                cache=cache,
                use_cache=True,
                use_causal_mask=True,
            ).logits
            ar_tokens = self._sample_tokens(
                verify_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            ar_token_ids = ar_tokens[0].tolist()
            block_ids = block[0].tolist()
            accepted = 1
            for i in range(block_length - 1):
                if ar_token_ids[i] == block_ids[i + 1]:
                    accepted += 1
                else:
                    break
            accepted = min(accepted, max_new_tokens - total_generated)
            accepted_tokens = ar_tokens[:, :accepted]

            self._trim_cache(cache, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            eos_index = _first_token_index(accepted_tokens[0], eos_token_ids)
            if eos_index is not None:
                accepted_tokens = accepted_tokens[:, : eos_index + 1]
            mx.eval(accepted_tokens)
            yield accepted_tokens
            total_generated += accepted_tokens.shape[1]
            if eos_index is not None:
                break

    def sanitize(self, weights):
        if self.config.tie_word_embeddings:
            weights.pop("diffusion_head.weight", None)

        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
            and not k.endswith(".self_attn.k_scale")
            and not k.endswith(".self_attn.v_scale")
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        return [KVCache() for _ in self.layers]
