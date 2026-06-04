"""Realtime, incremental multimodal sessions for MLX-VLM omni models.

Turn-based `generate()` waits for the *whole* utterance, then prefills it. A
`RealtimeSession` instead appends perception into ONE persistent KV cache as it
arrives: audio is ingested in ~200 ms micro-turns *during* speech, so the reply
starts the instant the user stops — no end-of-utterance re-prefill.

This exploits Gemma 4's encoder-free audio: a 200 ms chunk is just N×640 raw
samples → one linear projection → N tokens appended to the cache (no encoder, no
lookback). Measured on an M4 Max (gemma-4-12B-it 4-bit): ~4× realtime ingest,
~110 ms time-to-first-token.

Usage:
    sess = RealtimeSession(model, processor)
    sess.start_turn()
    for chunk in mic_200ms_chunks():     # float32 PCM @ 16 kHz
        sess.append_audio(chunk)
    sess.append_image(frame)             # optional: image goes before the audio
    for piece in sess.respond():
        print(piece, end="", flush=True)
"""

from __future__ import annotations

from typing import Callable, Iterator, Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .prompt_utils import apply_chat_template


def _last_logits(out) -> mx.array:
    logits = out.logits if hasattr(out, "logits") else out
    return logits[:, -1, :]


class RealtimeSession:
    """One persistent-cache, incrementally-fed multimodal conversation."""

    def __init__(
        self,
        model,
        processor,
        *,
        instruction: str = "Reply briefly and naturally.",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ):
        self.model, self.processor, self.tok = model, processor, processor.tokenizer
        cfg = model.config
        self.audio_token = int(cfg.audio_token_id)
        self.image_token = int(cfg.image_token_id)
        eos = getattr(cfg, "eos_token_id", None) or getattr(
            self.tok, "eos_token_id", None
        )
        self.stop_ids = (
            set(eos)
            if isinstance(eos, (list, tuple))
            else ({int(eos)} if eos is not None else set())
        )
        self.sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        self.bos = getattr(cfg, "bos_token_id", None) or getattr(
            self.tok, "bos_token_id", None
        )
        fe = getattr(processor, "feature_extractor", None)
        self.spt = int(
            getattr(
                fe,
                "audio_samples_per_token",
                getattr(cfg, "audio_samples_per_token", 640),
            )
        )

        # Derive the turn scaffold from the real chat template, split around the
        # audio-placeholder run. <bos> belongs once at the start, not per turn.
        ids = self._encode(
            apply_chat_template(processor, cfg, instruction, num_audios=1)
        )
        a = [i for i, t in enumerate(ids) if t == self.audio_token]
        pre, self.audio_suffix = ids[: a[0]], ids[a[-1] + 1 :]
        self.audio_prefix = (
            pre[1:] if (self.bos is not None and pre and pre[0] == self.bos) else pre
        )

        # Image+audio layout (image first), for sessions that show + ask.
        idav = self._encode(
            apply_chat_template(
                processor, cfg, "Here's my question:", num_images=1, num_audios=1
            )
        )
        ip = [i for i, t in enumerate(idav) if t == self.image_token]
        ap = [i for i, t in enumerate(idav) if t == self.audio_token]
        av_pre = idav[: ip[0]]
        self.av_pre = (
            av_pre[1:]
            if (self.bos is not None and av_pre and av_pre[0] == self.bos)
            else av_pre
        )
        self.av_mid = idav[ip[-1] + 1 : ap[0]]
        self.reset()

    # ------------------------------------------------------------------ core
    def _encode(self, templ) -> list[int]:
        return list(self.tok.encode(templ) if isinstance(templ, str) else templ)

    def reset(self) -> None:
        self.cache = make_prompt_cache(self.model.language_model)
        self._rem = np.zeros(0, dtype="float32")
        if self.bos is not None:
            self._feed([self.bos])

    def _sync(self) -> None:  # MLX is lazy; force the cache update to run now
        mx.eval([c.state for c in self.cache])

    def _feed(self, ids: list[int]) -> Optional[mx.array]:
        if not ids:
            return None
        logits = _last_logits(self.model(input_ids=mx.array([ids]), cache=self.cache))
        self._sync()
        return logits

    # -------------------------------------------------------------- ingestion
    def start_turn(self, image=None) -> None:
        """Open a user turn. With an image it leads the turn (image→text→audio),
        matching Gemma's intended layout; audio-only uses the audio scaffold."""
        if image is not None:
            self._feed_image(self.av_pre, image, self.av_mid)
        else:
            self._feed(self.audio_prefix)

    def append_audio(self, pcm) -> None:
        """Append float32 PCM @16 kHz (carries a <one-token remainder)."""
        buf = (
            np.concatenate([self._rem, pcm])
            if self._rem.size
            else np.asarray(pcm, "float32")
        )
        n = len(buf) // self.spt
        self._rem = buf[n * self.spt :]
        if not n:
            return
        feats = buf[: n * self.spt].reshape(1, n, self.spt).astype("float32")
        self.model(
            input_ids=mx.array([[self.audio_token] * n]),
            input_features=mx.array(feats),
            input_features_mask=mx.ones((1, n), mx.bool_),
            cache=self.cache,
        )
        self._sync()

    def respond(
        self,
        *,
        max_tokens: int = 200,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> Iterator[str]:
        """Close the turn and stream the reply token-by-token."""
        if self._rem.size:  # flush sub-token audio tail
            pad = np.zeros(self.spt, "float32")
            pad[: self._rem.size] = self._rem
            self._rem = np.zeros(0, "float32")
            self.append_audio(pad)
        logits = self._feed(self.audio_suffix)
        for _ in range(max_tokens):
            tid = int(self.sampler(logits).item())
            if tid in self.stop_ids:
                break
            yield self.tok.decode([tid])
            if should_abort is not None and should_abort():
                break
            logits = self._feed([tid])

    # ------------------------------------------------------------- image block
    def _feed_image(self, pre_ids, image, mid_ids) -> None:
        """Feed <pre><boi>[image]<eoi><mid> in ONE call, using the processor's
        real image block (ids + mm_token_type_ids). Splitting it across calls, or
        fabricating mm_token_type_ids, makes the model ignore the image."""
        proc = self.processor(
            text=[self.tok.decode([self.image_token])],
            images=[image],
            return_tensors="np",
        )
        pid = list(np.array(proc["input_ids"][0]).tolist())
        pmm = (
            list(np.array(proc["mm_token_type_ids"][0]).tolist())
            if proc.get("mm_token_type_ids") is not None
            else [0] * len(pid)
        )
        if pid and self.bos is not None and pid[0] == self.bos:
            pid, pmm = pid[1:], pmm[1:]
        pre = pre_ids[:-1] if (pre_ids and pid and pre_ids[-1] == pid[0]) else pre_ids
        mid = mid_ids[1:] if (mid_ids and pid and mid_ids[0] == pid[-1]) else mid_ids
        full = list(pre) + pid + list(mid)
        kw = {
            "input_ids": mx.array([full]),
            "pixel_values": mx.array(proc["pixel_values"]),
            "mm_token_type_ids": mx.array([[0] * len(pre) + pmm + [0] * len(mid)]),
            "cache": self.cache,
        }
        if proc.get("image_position_ids") is not None:
            kw["image_position_ids"] = mx.array(proc["image_position_ids"])
        self.model(**kw)
        self._sync()
