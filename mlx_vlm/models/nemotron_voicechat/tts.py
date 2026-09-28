# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""MLX inference modules for Nemotron VoiceChat's EAR-TTS decoder.

The module hierarchy intentionally mirrors the NeMo checkpoint.  Besides making
conversion mechanical, this lets mlx-vlm use the module-specific quantization map
already present in community MLX checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_audio.codec.models.nemotron_voicechat import NemotronVoiceChatCodec

from ..cache import KVCache, RotatingKVCache
from ..gemma3.config import TextConfig as Gemma3TextConfig
from ..gemma3.language import Gemma3Model
from .config import CharacterEncoderConfig, CodecConfig, MoGConfig, TTSConfig


class OffsetRMSNorm(nn.Module):
    """Gemma-style RMSNorm whose learned weight is an offset from one."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = mx.fast.rms_norm(x, 1.0 + self.weight.astype(mx.float32), self.eps)
        return x.astype(dtype)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class MLPLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.pre_norm = OffsetRMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = OffsetRMSNorm(hidden_size, eps)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.post_norm(self.mlp(self.pre_norm(x)))


def _top_p_logits(logits: mx.array, top_p: float) -> mx.array:
    if top_p >= 1.0:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    keep_sorted = cumulative > (1.0 - top_p)
    keep = mx.put_along_axis(mx.zeros_like(keep_sorted), indices, keep_sorted, axis=-1)
    return mx.where(keep, logits, -float("inf"))


class MoGHead(nn.Module):
    """Mixture-of-Gaussians head used for continuous RVQ refinement."""

    def __init__(self, hidden_size: int, out_size: int, config: MoGConfig):
        super().__init__()
        self.out_size = out_size
        self.low_rank = config.low_rank
        self.num_predictions = config.num_predictions
        self.min_log_std = config.min_log_std
        self.mlp_stack = [
            *[
                MLPLayer(
                    hidden_size,
                    config.intermediate_size,
                    config.eps,
                )
                for _ in range(config.num_layers)
            ],
            OffsetRMSNorm(hidden_size, config.eps),
        ]
        self.proj_logits = nn.Linear(hidden_size, config.num_predictions, bias=False)
        self.proj_mus = nn.Linear(
            hidden_size,
            config.num_predictions * config.low_rank,
            bias=False,
        )
        self.proj_logs = nn.Linear(hidden_size, 1, bias=False)
        self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
        self.low_mat = mx.zeros((config.num_predictions, out_size, config.low_rank))

    def infer(
        self,
        x: mx.array,
        *,
        guidance_scale: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[mx.array, mx.array]:
        for layer in self.mlp_stack:
            x = layer(x)

        if guidance_scale > 0.0:
            if x.shape[0] % 2:
                raise ValueError("classifier-free guidance requires an even batch")
            half = x.shape[0] // 2
            cond, uncond = x[:half], x[half:]
            x = cond + guidance_scale * (cond - uncond)

        b, t, _ = x.shape
        logits = _top_p_logits(self.proj_logits(x), top_p)
        uniform = mx.random.uniform(shape=logits.shape)
        gumbel = -mx.log(-mx.log(uniform + 1e-8) + 1e-8)
        component = mx.argmax(
            mx.log(mx.softmax(logits.astype(mx.float32), axis=-1)) + gumbel,
            axis=-1,
        )

        flat_x = x.reshape(-1, x.shape[-1])
        flat_component = component.reshape(-1)
        mus = self.proj_mus.weight.reshape(self.num_predictions, self.low_rank, -1)[
            flat_component
        ]
        mu = mx.matmul(mus, flat_x[..., None]).squeeze(-1)
        low_mat = self.low_mat[flat_component]
        mu = mx.matmul(low_mat, mu[..., None]).squeeze(-1).reshape(b, t, self.out_size)
        residual = self.proj_else(x)
        logs = mx.maximum(self.proj_logs(x), self.min_log_std)
        return mu * mx.exp(logs) + residual, logs


class T5GemmaSelfAttention(nn.Module):
    def __init__(self, config: CharacterEncoderConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = config.query_pre_attn_scalar**-0.5
        self.softcap = config.attn_logit_softcapping
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_base)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        b, length, _ = x.shape
        q = self.q_proj(x).reshape(b, length, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        if self.repeats > 1:
            k = mx.repeat(k, self.repeats, axis=1)
            v = mx.repeat(v, self.repeats, axis=1)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        scores = mx.tanh(scores / self.softcap) * self.softcap
        if mask is not None:
            scores = mx.where(mask[:, None, None, :], scores, -1e30)
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = mx.matmul(probs, v).transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out)


class T5GemmaEncoderLayer(nn.Module):
    def __init__(self, config: CharacterEncoderConfig):
        super().__init__()
        self.self_attn = T5GemmaSelfAttention(config)
        self.pre_self_attn_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_self_attn_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.pre_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = OffsetRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        x = x + self.post_self_attn_layernorm(
            self.self_attn(self.pre_self_attn_layernorm(x), mask)
        )
        return x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )


class T5GemmaEncoder(nn.Module):
    def __init__(self, config: CharacterEncoderConfig):
        super().__init__()
        self.layers = [
            T5GemmaEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = OffsetRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.scale = config.hidden_size**0.5

    def __call__(self, inputs_embeds: mx.array, mask: mx.array) -> mx.array:
        x = inputs_embeds * mx.array(self.scale, dtype=inputs_embeds.dtype)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class T5GemmaBackbone(nn.Module):
    def __init__(self, config: CharacterEncoderConfig):
        super().__init__()
        self.encoder = T5GemmaEncoder(config)


class SubwordFlagEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.pad_tensor = mx.array(vocab_size, dtype=mx.int32)
        self.is_continuation = mx.zeros((vocab_size + 1,), dtype=mx.int32)
        self.cont_emb = nn.Embedding(2, hidden_size)

    def __call__(self, embeds: mx.array, token_ids: mx.array) -> mx.array:
        safe = mx.where(
            token_ids >= self.is_continuation.shape[0] - 1, self.pad_tensor, token_ids
        )
        return embeds + self.cont_emb(self.is_continuation[safe])


class BOSEOSEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # The source implementation stores max(vocab.values()), not vocabulary
        # length, as its custom OOV index.
        self.pad_tensor = mx.array(vocab_size - 1, dtype=mx.int32)
        self.special_flags = mx.zeros((vocab_size,), dtype=mx.int32)
        self.special_emb = nn.Embedding(3, hidden_size)

    def __call__(self, embeds: mx.array, token_ids: mx.array) -> mx.array:
        safe = mx.where(
            token_ids >= self.special_flags.shape[0], self.pad_tensor, token_ids
        )
        return embeds + self.special_emb(self.special_flags[safe])


class CharAwareSubwordEncoder(nn.Module):
    def __init__(
        self,
        config: CharacterEncoderConfig,
        out_size: int,
        *,
        vocab_size: int = 131_072,
    ):
        super().__init__()
        self.backbone = T5GemmaBackbone(config)
        self.embed_tokens = nn.Embedding(config.char_vocab_size, config.hidden_size)
        self.proj_embedding = nn.Linear(config.hidden_size, out_size, bias=False)
        self.subword_flag_emb = SubwordFlagEmbedding(vocab_size, config.hidden_size)
        self.bos_eos_emb = BOSEOSEmbedding(vocab_size, config.hidden_size)
        self.char_padding_idx = config.char_vocab_size - 1
        self._subword_to_chars: dict[int, tuple[int, ...]] | None = None

    def set_vocabulary(self, vocabulary: Mapping[str, int]) -> None:
        """Build the exact dense character vocabulary used by NeMo."""
        single = {token: idx for token, idx in vocabulary.items() if len(token) == 1}
        characters = sorted(single, key=single.__getitem__)
        char_to_id = {char: idx for idx, char in enumerate(characters)}
        if len(char_to_id) + 1 != self.embed_tokens.weight.shape[0]:
            raise ValueError(
                "tokenizer-derived character vocabulary has "
                f"{len(char_to_id) + 1} entries, expected "
                f"{self.embed_tokens.weight.shape[0]}"
            )
        self._subword_to_chars = {
            idx: tuple(char_to_id[c] for c in token if c in char_to_id)
            for token, idx in vocabulary.items()
        }

    def _prepare_chars(
        self, subword_ids: mx.array, subword_mask: mx.array
    ) -> tuple[mx.array, mx.array, list[tuple[int, int]]]:
        if self._subword_to_chars is None:
            raise RuntimeError(
                "set_vocabulary(tokenizer.get_vocab()) must be called before TTS"
            )
        ids = subword_ids.tolist()
        valid = subword_mask.tolist()
        sequences: list[tuple[int, ...]] = []
        positions: list[tuple[int, int]] = []
        for batch_idx, (row, row_mask) in enumerate(zip(ids, valid)):
            for time_idx, (token, keep) in enumerate(zip(row, row_mask)):
                if keep:
                    sequences.append(self._subword_to_chars.get(int(token), ()))
                    positions.append((batch_idx, time_idx))

        max_length = max((len(sequence) for sequence in sequences), default=0)
        if not sequences or max_length == 0:
            return (
                mx.full((len(sequences), 1), self.char_padding_idx, mx.int32),
                mx.zeros((len(sequences),), mx.int32),
                positions,
            )
        char_ids = [
            list(sequence) + [self.char_padding_idx] * (max_length - len(sequence))
            for sequence in sequences
        ]
        lengths = [len(sequence) for sequence in sequences]
        return mx.array(char_ids, mx.int32), mx.array(lengths, mx.int32), positions

    def __call__(
        self,
        subword_ids: mx.array,
        subword_mask: mx.array | None = None,
    ) -> mx.array:
        if subword_mask is None:
            subword_mask = mx.ones(subword_ids.shape, dtype=mx.bool_)
        char_ids, lengths, positions = self._prepare_chars(subword_ids, subword_mask)
        out = mx.zeros(
            (*subword_ids.shape, self.proj_embedding.weight.shape[0]),
            dtype=self.embed_tokens.weight.dtype,
        )
        if positions and char_ids.shape[0] and char_ids.shape[1]:
            mask = mx.arange(char_ids.shape[1])[None, :] < lengths[:, None]
            hidden = self.backbone.encoder(self.embed_tokens(char_ids), mask)
            pooled = mx.sum(hidden * mask[..., None], axis=1) / mx.maximum(
                lengths[:, None], 1
            )
            projected = self.proj_embedding(pooled)
            batch_indices = mx.array([p[0] for p in positions], mx.int32)
            time_indices = mx.array([p[1] for p in positions], mx.int32)
            out = out.at[batch_indices, time_indices].add(projected)
        out = self.subword_flag_emb(out, subword_ids)
        return self.bos_eos_emb(out, subword_ids)


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, num_codebooks: int, eps: float):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = mx.zeros((hidden_size,), dtype=mx.float32)
        self.residual_scale = mx.array(0.5, dtype=mx.float32)
        self.final_norm = OffsetRMSNorm(hidden_size, eps)

    def __call__(self, audio: mx.array, text: mx.array) -> mx.array:
        audio = self.audio_proj(audio / self.num_codebooks)
        text = self.text_proj(text)
        gate = mx.sigmoid(self.gate).astype(audio.dtype)
        scale = mx.sigmoid(self.residual_scale).astype(audio.dtype)
        return self.final_norm(scale * (gate * audio + (1.0 - gate) * text))


def _gemma_config(config: TTSConfig) -> Gemma3TextConfig:
    return Gemma3TextConfig(
        model_type="gemma3_text",
        vocab_size=1,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        query_pre_attn_scalar=config.query_pre_attn_scalar,
        sliding_window=config.sliding_window,
        sliding_window_pattern=config.sliding_window_pattern,
        rope_global_base_freq=config.rope_global_base_freq,
        rope_local_base_freq=config.rope_local_base_freq,
    )


@dataclass
class TTSStepOutput:
    codes: mx.array
    cache: Sequence[KVCache | RotatingKVCache]
    hidden_states: mx.array


class RVQEARTTSModel(nn.Module):
    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config
        self.backbone = Gemma3Model(_gemma_config(config), scale_inputs_embeds=False)
        del self.backbone.embed_tokens
        self.bos_emb = mx.zeros((config.hidden_size,))
        self.null_emb = mx.zeros((config.hidden_size,))
        self.embed_code = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.embed_subword = CharAwareSubwordEncoder(
            config.character_encoder, config.hidden_size
        )
        self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
            config.hidden_size, config.num_quantizers, config.rms_norm_eps
        )
        self.audio_prompt_projection_W = mx.zeros(
            (config.hidden_size, config.hidden_size)
        )
        self.mog_head = MoGHead(config.hidden_size, config.latent_size, config.mog_head)
        self.rvq_embs = mx.zeros(
            (config.num_quantizers, config.codebook_size, config.latent_size)
        )

    def set_vocabulary(self, vocabulary: Mapping[str, int]) -> None:
        self.embed_subword.set_vocabulary(vocabulary)

    def make_cache(self):
        caches = []
        for idx in range(self.config.num_hidden_layers):
            if (idx + 1) % self.config.sliding_window_pattern == 0:
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                )
        return caches

    def depthsum_embedding(self, code: mx.array) -> mx.array:
        result = mx.zeros((*code.shape[:-1], self.config.latent_size))
        padding = mx.zeros(
            (self.config.num_quantizers, 1, self.config.latent_size),
            dtype=self.rvq_embs.dtype,
        )
        embeddings = mx.concatenate([self.rvq_embs, padding], axis=1)
        for idx in range(self.config.num_quantizers):
            result = result + embeddings[idx, code[..., idx]]
        return result

    def _condition(
        self,
        subword_ids: mx.array,
        subword_mask: mx.array,
        batch_size: int,
        guidance: bool,
    ) -> mx.array:
        cond = self.embed_subword(subword_ids, subword_mask)
        if guidance:
            cond = mx.concatenate([cond, cond], axis=0)
            null = mx.broadcast_to(self.null_emb, cond[batch_size:].shape)
            cond = mx.concatenate([cond[:batch_size], null], axis=0)
        return cond

    def warmup(
        self,
        code: mx.array,
        subword_ids: mx.array,
        subword_mask: mx.array,
        audio_mask: mx.array,
        audio_prompt_latent: mx.array,
        *,
        guidance: bool = True,
        cache=None,
    ) -> tuple[mx.array, Sequence[KVCache | RotatingKVCache]]:
        shifted = mx.concatenate([mx.zeros_like(code[:, :1]), code[:, :-1]], axis=1)
        code_embed = self.embed_code(self.depthsum_embedding(shifted))
        previous_audio = mx.concatenate(
            [mx.zeros_like(audio_mask[:, :1]), audio_mask[:, :-1]], axis=1
        )
        bos_mask = audio_mask & ~previous_audio
        pre_bos = mx.cumsum(bos_mask.astype(mx.int32), axis=1) == 0
        projected_prompt = mx.matmul(code_embed, self.audio_prompt_projection_W)
        if audio_prompt_latent is not None:
            projected_prompt = audio_prompt_latent.astype(code_embed.dtype)
        code_embed = mx.where(pre_bos[..., None], projected_prompt, code_embed)
        code_embed = code_embed + bos_mask[..., None] * self.bos_emb

        batch_size = code.shape[0]
        if guidance:
            code_embed = mx.concatenate([code_embed, code_embed], axis=0)
        cond = self._condition(
            subword_ids,
            subword_mask,
            batch_size,
            guidance,
        )
        inputs = self.gated_fusion_audio_text(code_embed, cond)
        if cache is None:
            cache = self.make_cache()
        hidden = self.backbone(None, inputs_embeds=inputs, cache=cache)
        return hidden, cache

    def _rvq_encode_step(
        self,
        residual: mx.array,
        code: mx.array,
        start: int,
        count: int,
    ) -> mx.array:
        pieces = [code[..., i] for i in range(self.config.num_quantizers)]
        for idx in range(start, start + count):
            embedding = self.rvq_embs[idx]
            distances = mx.sum(embedding * embedding, axis=-1) - 2.0 * mx.matmul(
                residual, embedding.T
            )
            selected = mx.argmin(distances, axis=-1)
            residual = residual - embedding[selected]
            pieces[idx] = selected
        return mx.stack(pieces, axis=-1)

    def generate_codes(self, hidden_states: mx.array) -> mx.array:
        config = self.config
        if hidden_states.shape[0] % 2:
            raise ValueError(
                "guided generation expects conditional/unconditional pairs"
            )
        half = hidden_states.shape[0] // 2
        conditional, unconditional = hidden_states[:half], hidden_states[half:]
        code = mx.full(
            (*conditional.shape[:2], config.num_quantizers),
            config.codebook_size,
            dtype=mx.int32,
        )
        rates = [i / config.num_iterations for i in range(config.num_iterations)]
        masked = [
            math.ceil(
                ((1.0 - rate**config.exponent) ** (1.0 / config.exponent))
                * config.num_quantizers
            )
            for rate in rates
        ]
        counts = [
            masked[i] - (masked[i + 1] if i + 1 < len(masked) else 0)
            for i in range(len(masked))
        ]

        completed = 0
        for count in counts:
            if count == 0:
                continue
            embedded = self.embed_code(self.depthsum_embedding(code))
            mog_input = mx.concatenate(
                [embedded + conditional, embedded + unconditional], axis=0
            )
            mu, logs = self.mog_head.infer(
                mog_input,
                guidance_scale=config.guidance_scale,
                top_p=config.top_p,
            )
            residual = (
                mu + mx.exp(logs) * mx.random.normal(mu.shape) * config.noise_scale
            )
            code = self._rvq_encode_step(residual, code, completed, count)
            completed += count
        return code

    def step(
        self,
        code: mx.array,
        subword_ids: mx.array,
        subword_mask: mx.array,
        cache,
        *,
        guidance: bool = True,
    ) -> TTSStepOutput:
        code_embed = self.embed_code(self.depthsum_embedding(code))
        batch_size = code.shape[0]
        if guidance:
            code_embed = mx.concatenate([code_embed, code_embed], axis=0)
        cond = self._condition(subword_ids, subword_mask, batch_size, guidance)
        inputs = self.gated_fusion_audio_text(code_embed, cond)
        hidden = self.backbone(None, inputs_embeds=inputs, cache=cache)
        return TTSStepOutput(
            codes=self.generate_codes(hidden), cache=cache, hidden_states=hidden
        )


class PromptLatents(nn.Module):
    def __init__(self, hidden_size: int, frames: int = 37):
        super().__init__()
        self.Aria = mx.zeros((1, frames, hidden_size))


class SpeechDecoder(nn.Module):
    """Checkpoint-compatible wrapper around codec, EAR-TTS, and prompt buffers."""

    def __init__(self, tts_config: TTSConfig, codec_config: CodecConfig):
        super().__init__()
        self.audio_codec = NemotronVoiceChatCodec(codec_config)
        self.tts_model = RVQEARTTSModel(tts_config)
        self.control_codes = mx.zeros((3,), dtype=mx.int32)
        self.audio_prompt_latents = PromptLatents(tts_config.hidden_size)
        self.codec_silence_tokens = mx.zeros(
            (tts_config.num_quantizers,), dtype=mx.int32
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        codec_weights = {}
        for key, value in weights.items():
            if key == "_control_codes":
                sanitized["control_codes"] = value
            elif key.startswith("audio_codec."):
                codec_weights[key[len("audio_codec.") :]] = value
            else:
                sanitized[key] = value
        for key, value in self.audio_codec.sanitize(codec_weights).items():
            sanitized[f"audio_codec.{key}"] = value
        return sanitized
