"""Youtu-Parsing-Omni top-level model: omni encoder + MLA language model.

Ported from tencent/Youtu-Parsing-Omni (https://huggingface.co/tencent/Youtu-Parsing-Omni).
Copyright (C) 2026 Tencent. The original code is distributed under the Youtu-Parsing
license (https://huggingface.co/tencent/Youtu-Parsing-Omni/blob/main/LICENSE);
modified / re-implemented here for MLX.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .omni import OmniModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.omni_model = OmniModel(config.omni_config)
        self.language_model = LanguageModel(config.text_config, config)

    def _scatter(self, out, flat, feats, what):
        if flat.shape[0] != feats.shape[0]:
            raise ValueError(
                f"Number of {what} positions ({flat.shape[0]}) does not match "
                f"number of {what} features ({feats.shape[0]})"
            )
        out[mx.array(flat)] = feats
        return out

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if kwargs.get("pixel_values_videos") is not None:
            raise NotImplementedError("Video input is not supported yet.")
        audios = kwargs.get("audios")
        if audios is None:
            audios = kwargs.get("input_features")

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        video_images = kwargs.get("video_images")
        if pixel_values is None and audios is None and video_images is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        B, L, D = inputs_embeds.shape
        out = inputs_embeds.reshape(B * L, D)

        if pixel_values is not None:
            grid_thw = kwargs.get("image_grid_thw")
            if grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            grid_thw = np.array(grid_thw)

            # encoder runs in its own (default float32) dtype; merged features are cast
            # to the LM dtype only here, at the scatter.
            feats = self.omni_model.forward_vision(pixel_values, grid_thw)
            feats = feats.astype(inputs_embeds.dtype)

            image_indices = kwargs.get("image_indices")
            if image_indices is not None:
                idx = np.array(image_indices)
                flat = idx[0].reshape(-1) * L + idx[1].reshape(-1)
            else:
                flat = np.nonzero(
                    np.array(input_ids == self.config.image_token_id).reshape(-1)
                )[0]
            out = self._scatter(out, flat, feats, "image")

        if audios is not None:
            if isinstance(audios, (mx.array, np.ndarray)) and audios.ndim == 2:
                audios = [audios]  # a single clip
            elif isinstance(audios, (mx.array, np.ndarray)):
                audios = list(audios)  # stacked [N, T, mel] (equal-length clips)
            feats = mx.concatenate(self.omni_model.forward_audio(list(audios)), axis=0)
            feats = feats.astype(inputs_embeds.dtype)

            audio_indices = kwargs.get("audio_indices")
            if audio_indices is not None:
                parts = (
                    list(audio_indices)
                    if isinstance(audio_indices, (list, tuple))
                    else (
                        list(audio_indices)
                        if np.ndim(audio_indices) == 4
                        else [audio_indices]
                    )
                )
                parts = [
                    np.array(q) for q in parts
                ]  # each [2, 1, n]: (batch, position)
                flat = np.concatenate(
                    [q[0].reshape(-1) * L + q[1].reshape(-1) for q in parts]
                )
            else:
                flat = np.nonzero(
                    np.array(input_ids == self.config.audio_token_id).reshape(-1)
                )[0]
            out = self._scatter(out, flat, feats, "audio")

        if video_images is not None:
            out = self._video_embeddings(
                out, L, inputs_embeds.dtype, video_images, kwargs
            )

        return InputEmbeddingsFeatures(inputs_embeds=out.reshape(B, L, D))

    def _video_embeddings(self, out, L, dtype, video_images, kw):
        """Joint video pass (frames + audio track share the omni encoder), scattered into ``out``."""
        grid = kw.get("video_image_grid_thw")
        split = kw.get("video_split")
        img_idx = kw.get("video_image_indices")
        if grid is None or split is None or img_idx is None:
            raise ValueError(
                "video_images needs video_image_grid_thw, video_split and video_image_indices"
            )
        img_idx = np.array(img_idx)
        video_audios = kw.get("video_audios")
        aud_idx = kw.get("video_audio_indices")
        has_audio = video_audios is not None and len(video_audios) > 0
        if has_audio and aud_idx is None:
            raise ValueError("video_audios needs video_audio_indices")
        aud_parts = (
            [np.array(q) for q in aud_idx] if has_audio else []
        )  # each [2, 1, n]
        if has_audio and not isinstance(video_audios, (list, tuple)):
            video_audios = list(video_audios)
        video_images = (
            video_images
            if isinstance(video_images, mx.array)
            else mx.array(video_images)
        )
        if self.config.video_omni_fusion:
            vis, auds = self.omni_model.forward_video(
                video_images,
                np.array(grid),
                video_audios if has_audio else None,
                np.array(split),
                frame_seq_pos=img_idx[1].reshape(-1).tolist(),
                chunk_seq_pos=(
                    [int(q[1].reshape(-1)[0]) for q in aud_parts] if has_audio else None
                ),
            )
        else:  # HF video_omni_fusion=False: frames / audio chunks go through the plain image / audio passes
            vis = self.omni_model.forward_vision(video_images, np.array(grid))
            auds = (
                self.omni_model.forward_audio(list(video_audios)) if has_audio else []
            )
        flat = img_idx[0].reshape(-1) * L + img_idx[1].reshape(-1)
        out = self._scatter(out, flat, vis.astype(dtype), "video frame")
        if has_audio:
            flat = np.concatenate(
                [q[0].reshape(-1) * L + q[1].reshape(-1) for q in aud_parts]
            )
            out = self._scatter(
                out, flat, mx.concatenate(auds, axis=0).astype(dtype), "video audio"
            )
        return out

    @property
    def quant_predicate(self):
        # never quantize the (precision sensitive) omni encoder
        return lambda path, _: not path.startswith("omni_model.")

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, feats.inputs_embeds, mask=mask, cache=cache
        )

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith("model.omni_model."):
                nk = k[len("model.") :]
                # torch Conv2d [O, I, kH, kW] -> MLX [O, kH, kW, I] (skip if already converted)
                if "audio_embeddings.conv2d" in nk and nk.endswith(".weight"):
                    if v.ndim == 4 and v.shape[-2:] == (3, 3):
                        v = v.transpose(0, 2, 3, 1)
            elif k.startswith("model.language_model."):
                nk = "language_model.model." + k[len("model.language_model.") :]
            elif k.startswith("lm_head."):
                nk = "language_model." + k
            elif k.startswith("model.vision_model.") or k.startswith(
                "model.audio_model."
            ):
                continue
            else:
                nk = k
            if "rotary_emb.inv_freq" in nk:
                continue
            out[nk] = v

        # Keep the omni encoder (patch embed + layers + merger) in encoder_dtype
        # (float32 by default); the LM keeps the checkpoint / converted dtype.
        enc_dt = self.config.encoder_dtype
        if enc_dt and enc_dt != "same":
            enc_dt = getattr(mx, enc_dt)
            for k, v in out.items():
                if k.startswith("omni_model.") and mx.issubdtype(v.dtype, mx.floating):
                    out[k] = v.astype(enc_dt)

        if self.config.text_config.tie_word_embeddings:
            out.pop("language_model.lm_head.weight", None)

        # split kv_b_proj [H*(nope+v), r] into per-head embed_q (k) / unembed_out (v)
        t = self.config.text_config
        H, nope, vd = t.num_attention_heads, t.qk_nope_head_dim, t.v_head_dim
        for i in range(t.num_hidden_layers):
            p = f"language_model.model.layers.{i}.self_attn"
            key = f"{p}.kv_b_proj.weight"
            if key not in out:
                continue
            w = out.pop(key).reshape(H, nope + vd, -1)
            out[f"{p}.embed_q.weight"] = mx.contiguous(w[:, :nope, :].swapaxes(-1, -2))
            out[f"{p}.unembed_out.weight"] = mx.contiguous(w[:, nope:, :])
        return out
