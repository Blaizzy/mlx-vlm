from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.llama import LlamaModel, ModelArgs
from mlx_lm.sample_utils import apply_top_k, apply_top_p

from .config import MiniCPMTTSConfig


def materialize_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    if g.ndim == 1:
        g = g[:, None]
    v_float = v.astype(mx.float32)
    denom = mx.sqrt(mx.sum(v_float * v_float, axis=1, keepdims=True))
    denom = mx.maximum(denom, mx.array(1e-12, dtype=denom.dtype))
    return v * (g / denom.astype(g.dtype))


def sanitize_tts_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    sanitized = {}
    head_g = {}
    head_v = {}

    for key, value in weights.items():
        if key.startswith("tts.head_code.") and ".parametrizations.weight." in key:
            parts = key.split(".")
            if len(parts) >= 6:
                head_idx = parts[2]
                original = parts[-1]
                if original == "original0":
                    head_g[head_idx] = value
                elif original == "original1":
                    head_v[head_idx] = value
            continue
        if "rotary_emb.inv_freq" in key:
            continue
        if key.startswith("tts."):
            sanitized[key] = value

    for head_idx, v in head_v.items():
        g = head_g.get(head_idx)
        if g is None:
            continue
        sanitized[f"tts.head_code.{head_idx}.weight"] = materialize_weight_norm(g, v)

    return sanitized


@dataclass
class TTSSamplingParams:
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 25
    repetition_penalty: Optional[float] = 1.05
    temperature: float = 0.8
    repetition_context_size: int = 16


@dataclass
class MiniCPMTTSGenerationOutput:
    new_ids: mx.array
    finished: bool


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.relu(self.linear1(x)))


class MiniCPMMLP(nn.Module):
    def __init__(self, config: MiniCPMTTSConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size)
        self.up_proj = nn.Linear(config.llm_hidden_size, config.llm_intermediate_size)
        self.down_proj = nn.Linear(config.llm_intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMTTS(nn.Module):
    def __init__(self, config: MiniCPMTTSConfig):
        super().__init__()
        self.config = config
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.use_text = config.use_text
        self.streaming = config.streaming
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty
        self.condition_type = config.condition_type

        if config.backbone_model != "llama":
            raise ValueError(
                f"Unsupported MiniCPM-o TTS backbone: {config.backbone_model}"
            )

        llama_args = ModelArgs(
            model_type="llama",
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.backbone_vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            tie_word_embeddings=True,
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.model = LlamaModel(llama_args)
        self.projector_spk = self.create_projector(config)
        self.projector_semantic = self.create_projector(config)
        self.emb_code = [
            nn.Embedding(config.num_audio_tokens, config.hidden_size)
            for _ in range(config.num_vq)
        ]
        self.head_code = [
            nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False)
            for _ in range(config.num_vq)
        ]

    @staticmethod
    def create_projector(config: MiniCPMTTSConfig):
        if config.projector_type == "mlp":
            return MultiModalProjector(config.llm_dim, config.hidden_size)
        if config.projector_type == "minicpm":
            return MiniCPMMLP(config)
        if config.projector_type == "default":
            return nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        raise ValueError(
            f"Unsupported MiniCPM-o TTS projector: {config.projector_type}"
        )

    @staticmethod
    def _apply_repetition_penalty(
        logits: mx.array,
        generated: list[int],
        penalty: Optional[float],
        context_size: int,
    ) -> mx.array:
        if penalty is None or penalty == 1 or not generated:
            return logits

        recent = generated[-context_size:]
        counts: dict[int, int] = {}
        for token in recent:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            alpha = penalty**count
            selected = logits[:, token]
            selected = mx.where(selected < 0, selected * alpha, selected / alpha)
            logits[:, token] = selected
        return logits

    @staticmethod
    def _sample(logits: mx.array, params: TTSSamplingParams) -> mx.array:
        if params.temperature == 0:
            return mx.argmax(logits, axis=-1)

        logits = logits / params.temperature
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if params.top_p is not None and 0 < params.top_p < 1.0:
            logprobs = apply_top_p(logprobs, params.top_p)
        if params.top_k is not None and params.top_k > 0:
            top_k = min(int(params.top_k), logprobs.shape[-1])
            if top_k < logprobs.shape[-1]:
                logprobs = apply_top_k(logprobs, top_k)
        return mx.random.categorical(logprobs)

    def generate(
        self,
        inputs_embeds: mx.array,
        eos_token: Optional[int] = None,
        force_no_stop: bool = False,
        min_new_token: int = 50,
        max_new_token: int = 2048,
        sampling_params: Optional[TTSSamplingParams] = None,
    ) -> MiniCPMTTSGenerationOutput:
        if inputs_embeds.shape[0] != 1:
            raise ValueError("MiniCPM-o TTS generation currently supports batch size 1")

        if sampling_params is None:
            sampling_params = TTSSamplingParams(
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                temperature=self.config.temperature,
            )
        if eos_token is None:
            eos_token = self.config.num_audio_tokens - 1

        cache = [KVCache() for _ in self.model.layers]
        generated: list[list[int]] = [[] for _ in range(self.num_vq)]
        new_tokens = []
        finished = False
        current_embeds = inputs_embeds

        for step in range(max_new_token):
            if step > 0:
                code_embeds = []
                for q in range(self.num_vq):
                    prev = mx.array([[generated[q][-1]]], dtype=mx.int32)
                    code_embeds.append(self.emb_code[q](prev))
                current_embeds = mx.stack(code_embeds, axis=-1).sum(axis=-1)

            hidden = self.model(
                mx.zeros(current_embeds.shape[:2], dtype=mx.int32),
                cache=cache,
                input_embeddings=current_embeds,
            )

            logits_per_vq = []
            for q in range(self.num_vq):
                logits = self.head_code[q](hidden)[:, -1, :].astype(mx.float32)
                if step > 0:
                    logits = self._apply_repetition_penalty(
                        logits,
                        generated[q],
                        sampling_params.repetition_penalty,
                        sampling_params.repetition_context_size,
                    )
                if step < min_new_token or force_no_stop:
                    logits[:, eos_token] = -mx.inf
                logits_per_vq.append(logits)

            next_codes = []
            for q, logits in enumerate(logits_per_vq):
                token = self._sample(logits, sampling_params)
                token_id = int(token.item())
                generated[q].append(token_id)
                next_codes.append(token_id)

            new_tokens.append(next_codes)
            if eos_token in next_codes:
                finished = True
                break

        if len(new_tokens) == 0:
            ids = mx.zeros((1, 0, self.num_vq), dtype=mx.int32)
        else:
            ids = mx.array(new_tokens, dtype=mx.int32)[None, :, :]
        return MiniCPMTTSGenerationOutput(new_ids=ids, finished=finished)
