"""Offline, model-specific VoiceChat inference session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mlx.core as mx
from mlx_audio.stt.models.nemotron_asr import tokenizer as rnnt_tokenizer
from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram
from mlx_audio.stt.utils import load_audio

from .model import Model


@dataclass
class VoiceChatResult:
    text: str
    audio: mx.array
    sample_rate: int
    text_tokens: mx.array
    audio_codes: mx.array
    function_tokens: mx.array
    user_transcript: str | None = None


class VoiceChatSession:
    def __init__(self, model: Model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.tts_model.tts_model.set_vocabulary(tokenizer.get_vocab())

    def _decode_text(self, tokens: mx.array) -> str:
        special = {
            self.model.config.pad_token_id,
            self.model.config.silence_token_id,
            self.model.config.bos_token_id,
            self.model.config.eos_token_id,
        }
        ids = [int(token) for token in tokens.tolist() if int(token) not in special]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _rnnt_decode(self, encoded: mx.array, length: int) -> str | None:
        vocabulary = self.model.config.rnnt_vocabulary
        if not vocabulary:
            return None
        decoder = self.model.stt_model.rnnt_decoder
        joint = self.model.stt_model.rnnt_joint
        blank = self.model.config.rnnt_blank_id
        last_token = blank
        hidden = None
        result: list[int] = []
        time = 0
        new_symbols = 0
        while time < length:
            token = (
                mx.array([[last_token]], dtype=mx.int32)
                if last_token != blank
                else None
            )
            prediction, proposed_hidden = decoder(token, hidden)
            logits = joint(
                encoded[:, time : time + 1],
                prediction.astype(encoded.dtype),
            )
            next_token = int(mx.argmax(logits))
            if next_token == blank:
                time += 1
                new_symbols = 0
                continue
            last_token = next_token
            hidden = tuple(value.astype(encoded.dtype) for value in proposed_hidden)
            if not rnnt_tokenizer.is_special_token(next_token, vocabulary):
                result.append(next_token)
            new_symbols += 1
            if new_symbols >= self.model.config.audio_config.max_symbols:
                time += 1
                new_symbols = 0
        return rnnt_tokenizer.decode(result, vocabulary).strip()

    def _tts_prompt(self, batch_size: int = 1):
        tts = self.model.tts_model
        config = self.model.config.tts_config
        frames = tts.audio_prompt_latents.Aria.shape[1]
        total = frames + 1
        prompt_samples = total * tts.audio_codec.waveform_to_token_ratio
        codes = tts.audio_codec.encode(
            mx.zeros((batch_size, 1, prompt_samples), dtype=mx.float32)
        ).transpose(0, 2, 1)
        if codes.shape != (batch_size, total, config.num_quantizers):
            raise RuntimeError(
                "silent prompt codec produced unexpected shape "
                f"{codes.shape}, expected "
                f"{(batch_size, total, config.num_quantizers)}"
            )
        pieces = [codes[:, idx] for idx in range(total)]
        mask_codes = mx.full(
            (batch_size, config.num_quantizers),
            config.codebook_size,
            dtype=mx.int32,
        )
        pieces[0] = mask_codes
        pieces[-2] = mask_codes
        codes = mx.stack(pieces, axis=1)

        subwords = mx.full(
            (batch_size, frames),
            self.model.config.pad_token_id,
            dtype=mx.int32,
        )
        subword_mask = mx.zeros((batch_size, frames), dtype=mx.bool_)
        subword_mask = subword_mask.at[:, -2:].add(True)
        audio_mask = mx.zeros((batch_size, frames), dtype=mx.bool_)
        audio_mask = audio_mask.at[:, -1:].add(True)
        latent = mx.broadcast_to(
            tts.audio_prompt_latents.Aria[:1],
            (batch_size, frames, config.hidden_size),
        )
        return codes[:, :-1], subwords, subword_mask, audio_mask, latent

    def _replace_control_codes(self, codes: mx.array) -> mx.array:
        control = self.model.tts_model.control_codes
        mask = mx.zeros(codes.shape, dtype=mx.bool_)
        for token in control.tolist():
            mask = mask | (codes == int(token))
        silence = mx.broadcast_to(
            self.model.tts_model.codec_silence_tokens[None, None, :],
            codes.shape,
        )
        return mx.where(mask, silence, codes)

    def create_streaming_session(
        self,
        *,
        system_prompt: str | None = None,
        seed: int = 0,
        max_streaming_seconds: float | None = None,
        use_language_cache: bool = True,
        use_perception_cache: bool = True,
        profile: bool = False,
    ):
        """Create an independent cache-aware online inference session."""

        from .streaming import VoiceChatStreamingSession

        return VoiceChatStreamingSession(
            self,
            system_prompt=system_prompt,
            seed=seed,
            max_streaming_seconds=max_streaming_seconds,
            use_language_cache=use_language_cache,
            use_perception_cache=use_perception_cache,
            profile=profile,
        )

    def generate(
        self,
        audio: Union[str, Path, mx.array],
        *,
        system_prompt: str | None = None,
        max_frames: int | None = None,
        extra_decoding_seconds: float = 0.0,
        seed: int = 0,
        use_language_cache: bool = False,
    ) -> VoiceChatResult:
        """Run one offline audio turn using the cached Aria voice."""
        mx.random.seed(seed)
        if isinstance(audio, (str, Path)):
            waveform = load_audio(str(audio), sr=self.model.config.input_sample_rate)
        else:
            waveform = audio.astype(mx.float32)
        waveform = waveform.squeeze()
        if waveform.ndim != 1:
            raise ValueError("audio must be a mono waveform")
        if extra_decoding_seconds < 0:
            raise ValueError("extra_decoding_seconds must be non-negative")
        if extra_decoding_seconds:
            waveform = mx.pad(
                waveform,
                (
                    0,
                    round(extra_decoding_seconds * self.model.config.input_sample_rate),
                ),
            )

        mel = log_mel_spectrogram(waveform, self.model.config.audio_config.preprocessor)
        mel_lengths = mx.array([mel.shape[1]], dtype=mx.int32)
        audio_embeds, lengths, asr_embeds = self.model.stt_model.perception(
            mel, mel_lengths
        )
        audio_frames = int(lengths[0])
        if max_frames is not None:
            if max_frames < 2:
                raise ValueError("max_frames must be at least 2")
            audio_frames = min(audio_frames, max_frames)
        audio_embeds = audio_embeds[:, :audio_frames]
        asr_embeds = asr_embeds[:, :audio_frames]

        prompt_text = (
            self.model.config.default_system_prompt
            if system_prompt is None
            else system_prompt
        )
        prompt_ids: list[int] = []
        if prompt_text.strip():
            prompt_ids = [
                self.model.config.bos_token_id,
                *self.tokenizer.encode(
                    prompt_text,
                    add_special_tokens=False,
                ),
                self.model.config.eos_token_id,
            ]
            prompt_embeds = self.model.stt_model.embed_tokens(
                mx.array([prompt_ids], dtype=mx.int32)
            ).astype(audio_embeds.dtype)
            # NeMo prepends system-token embeddings on the user-audio channel.
            # Output channels remain PAD over this prefix and are trimmed below.
            audio_embeds = mx.concatenate([prompt_embeds, audio_embeds], axis=1)

        prompt_frames = len(prompt_ids)
        timeline_frames = prompt_frames + audio_frames
        mx.eval(audio_embeds, asr_embeds)

        pad_id = self.model.config.pad_token_id
        text_tokens = [pad_id] * timeline_frames
        function_tokens = [pad_id] * timeline_frames
        stt_cache = self.model.stt_model.make_cache() if use_language_cache else None
        input_history: list[mx.array] = []

        def language_step(inputs: mx.array):
            if stt_cache is not None:
                return self.model.stt_model(inputs, cache=stt_cache)
            input_history.append(inputs)
            return self.model.stt_model(mx.concatenate(input_history, axis=1))

        prompt = self._tts_prompt()
        _, tts_cache = self.model.tts_model.tts_model.warmup(
            *prompt[:-1],
            audio_prompt_latent=prompt[-1],
            guidance=True,
        )
        previous_code = prompt[0][:, -1:]
        first_context_subword = prompt[1][:, -1:]
        generated_codes = mx.zeros(
            (1, timeline_frames, self.model.config.tts_config.num_quantizers),
            dtype=mx.int32,
        )

        for time in range(timeline_frames):
            # NeMo's misleadingly named ``_get_bos_embedding`` initializes the
            # first agent position with PAD, not BOS.
            previous_text_id = pad_id if time == 0 else text_tokens[time - 1]
            previous_function_id = pad_id if time == 0 else function_tokens[time - 1]
            previous_text = mx.array([[previous_text_id]], dtype=mx.int32)
            previous_function = mx.array([[previous_function_id]], dtype=mx.int32)
            fused = (
                self.model.stt_model.embed_tokens(previous_text)
                + audio_embeds[:, time : time + 1]
                + self.model.config.function_channel_weight
                * self.model.stt_model.embed_tokens(previous_function)
            )
            output = language_step(fused)
            if time >= prompt_frames:
                text_tokens[time] = int(mx.argmax(output.text_logits[:, -1]))
                function_tokens[time] = int(mx.argmax(output.function_logits[:, -1]))

            # Like the reference loop, time zero warms the language model but
            # does not request a TTS code. Every later position, including a
            # system-prompt prefix, advances EAR-TTS state.
            if time == 0:
                continue

            current = mx.array([[text_tokens[time]]], dtype=mx.int32)
            if text_tokens[time] == self.model.config.eos_token_id:
                previous_code = mx.broadcast_to(
                    self.model.tts_model.codec_silence_tokens[None, None, :],
                    previous_code.shape,
                )
            previous_subword = (
                first_context_subword
                if time == 1
                else mx.array([[text_tokens[time - 1]]], dtype=mx.int32)
            )
            # Kept for protocol parity; context_hidden_size is disabled in this
            # checkpoint, so the previous subword is intentionally unused.
            del previous_subword
            tts_output = self.model.tts_model.tts_model.step(
                previous_code,
                current,
                mx.ones(current.shape, dtype=mx.bool_),
                tts_cache,
                guidance=True,
            )
            previous_code = tts_output.codes
            generated_codes = generated_codes.at[:, time : time + 1].add(previous_code)
            mx.eval(previous_code, output.text_logits, output.function_logits)

        text_tokens = text_tokens[prompt_frames:]
        function_tokens = function_tokens[prompt_frames:]
        generated_codes = generated_codes[:, prompt_frames:]
        generated_codes = self._replace_control_codes(generated_codes)
        decoded = self.model.tts_model.audio_codec.decode(
            generated_codes.transpose(0, 2, 1)
        )
        mx.eval(decoded)

        text_array = mx.array(text_tokens, dtype=mx.int32)
        function_array = mx.array(function_tokens, dtype=mx.int32)
        return VoiceChatResult(
            text=self._decode_text(text_array),
            audio=decoded[0, 0],
            sample_rate=self.model.config.output_sample_rate,
            text_tokens=text_array,
            audio_codes=generated_codes[0],
            function_tokens=function_array,
            user_transcript=self._rnnt_decode(asr_embeds, audio_frames),
        )
