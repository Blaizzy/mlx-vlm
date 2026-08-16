"""Checkpoint-compatible Nemotron VoiceChat model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_audio.stt.models.nemotron_asr.conformer import Conformer
from mlx_audio.stt.models.nemotron_asr.rnnt import JointNetwork, PredictNetwork

from ..cache import ArraysCache, KVCache
from ..nemotron_h.language import NemotronHModel
from .config import AudioConfig, ModelConfig
from .tts import SpeechDecoder


@dataclass
class VoiceChatOutput:
    hidden_states: mx.array
    text_logits: mx.array
    function_logits: mx.array
    cache: object | None = None


class Perception(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.encoder = Conformer(config.encoder)
        self.proj = nn.Linear(config.encoder.d_model, config.output_dim, bias=True)

    def __call__(
        self, mel: mx.array, lengths: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        encoded, lengths = self.encoder(
            mel, lengths=lengths, att_context_size=self.encoder.args.att_context_size[0]
        )
        return self.proj(encoded), lengths, encoded


class DuplexSTTModel(nn.Module):
    """Fused audio/text Nemotron-H and auxiliary RNNT branch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        vocab_size = config.text_config.vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.llm = NemotronHModel(config.text_config, with_embeddings=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.function_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.perception = Perception(config.audio_config)
        self.rnnt_decoder = PredictNetwork(config.audio_config.decoder)
        self.rnnt_joint = JointNetwork(config.audio_config.joint)

    def __call__(
        self,
        inputs_embeds: mx.array,
        *,
        cache=None,
    ) -> VoiceChatOutput:
        hidden = self.llm(inputs_embeds=inputs_embeds, cache=cache)
        return VoiceChatOutput(
            hidden_states=hidden,
            text_logits=self.lm_head(hidden),
            function_logits=self.function_head(hidden),
            cache=cache,
        )

    def make_cache(self):
        caches = []
        for layer in self.llm.layers:
            if layer.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif layer.block_type == "*":
                caches.append(KVCache())
        return caches


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.stt_model = DuplexSTTModel(config)
        self.tts_model = SpeechDecoder(config.tts_config, config.codec_config)

    def __call__(self, inputs_embeds: mx.array, cache=None, **kwargs):
        return self.stt_model(inputs_embeds, cache=cache)

    def create_session(self, processor):
        """Create the model-specific runtime from a processor returned by ``load``."""
        tokenizer = getattr(processor, "tokenizer", processor)
        required = ("decode", "encode", "get_vocab")
        missing = [name for name in required if not hasattr(tokenizer, name)]
        if missing:
            raise TypeError(
                "Nemotron VoiceChat requires a tokenizer-like processor with "
                + ", ".join(missing)
            )

        from .session import VoiceChatSession

        return VoiceChatSession(self, tokenizer)

    @property
    def layers(self):
        return self.stt_model.llm.layers

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Convert PyTorch convolution/LSTM layouts and private buffers."""
        converted: dict[str, mx.array] = {}
        lstm_biases: dict[str, list[mx.array]] = {}
        tts_weights: dict[str, mx.array] = {}

        for key, value in weights.items():
            if key.startswith("stt_model.perception.preprocessor."):
                # Deterministic mel-filter/window buffers are rebuilt by mlx-audio.
                continue
            if key.startswith("tts_model."):
                tts_weights[key[len("tts_model.") :]] = value
                continue

            if key.startswith("stt_model.rnnt_decoder.") and ".dec_rnn.lstm." in key:
                base, suffix = key.rsplit(".dec_rnn.lstm.", 1)
                stem = f"{base}.dec_rnn.lstm"
                if suffix.startswith("weight_ih_l"):
                    layer = suffix[len("weight_ih_l") :]
                    converted[f"{stem}.{layer}.Wx"] = value
                elif suffix.startswith("weight_hh_l"):
                    layer = suffix[len("weight_hh_l") :]
                    converted[f"{stem}.{layer}.Wh"] = value
                elif suffix.startswith(("bias_ih_l", "bias_hh_l")):
                    layer = suffix.rsplit("_l", 1)[1]
                    lstm_biases.setdefault(f"{stem}.{layer}.bias", []).append(value)
                continue

            if value.ndim == 4 and key.startswith("stt_model.perception."):
                value = value.transpose(0, 2, 3, 1)
            elif value.ndim == 3 and (
                key.startswith("stt_model.perception.")
                or "stt_model.llm." in key
                and ".conv1d.weight" in key
            ):
                value = value.transpose(0, 2, 1)
            converted[key] = value

        for key, values in lstm_biases.items():
            converted[key] = sum(values)
        for key, value in self.tts_model.sanitize(tts_weights).items():
            converted[f"tts_model.{key}"] = value
        return converted

    @property
    def cast_predicate(self):
        def predicate(key: str) -> bool:
            return "A_log" not in key and not key.endswith(
                ("special_flags", "is_continuation", "pad_tensor")
            )

        return predicate
