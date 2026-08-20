from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from .config import (
    IDEOGRAM_4_FP8_REPO_ID,
    Ideogram4Variant,
    get_variant,
    validate_dimensions,
)
from .download import download_model, validate_model_layout
from .latent_norm import get_latent_norm
from .prompting import NormalizedPrompt
from .prompting import prepare_prompt as prepare_ideogram_prompt
from .scheduler import get_preset, get_schedule_for_resolution, make_step_intervals
from .transformer import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from .weights import PRECISION, load_text_encoder, load_transformer, load_vae

QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
IMAGE_POSITION_OFFSET = 65536
PATCH_SIZE = 2
AE_SCALE_FACTOR = 8
LATENT_DIM = 128
MAX_TEXT_TOKENS = 2048


@dataclass(frozen=True, slots=True)
class Ideogram4RuntimeConfig:
    evict_text_encoder: bool = True
    evict_transformers: bool = False


class Ideogram4ImagePipeline:
    def __init__(
        self,
        *,
        variant: str | Ideogram4Variant = IDEOGRAM_4_FP8_REPO_ID,
        model_path: str | Path,
        runtime_config: Ideogram4RuntimeConfig | None = None,
    ) -> None:
        self.variant = get_variant(variant)
        self.model_path = validate_model_layout(model_path)
        self.runtime_config = runtime_config or Ideogram4RuntimeConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path / "tokenizer",
            local_files_only=True,
        )
        self.text_encoder = None
        self.conditional_transformer = None
        self.unconditional_transformer = None
        self.vae = None

    @classmethod
    def from_pretrained(
        cls,
        variant: str | Ideogram4Variant = IDEOGRAM_4_FP8_REPO_ID,
        *,
        model_path: str | Path | None = None,
        download: bool = True,
        token: str | None = None,
        revision: str | None = None,
        force_download: bool = False,
    ) -> "Ideogram4ImagePipeline":
        spec = get_variant(variant)
        if model_path is None:
            if not download:
                raise FileNotFoundError(
                    f"No local model_path was provided for {spec.repo_id}"
                )
            model_path = download_model(
                spec,
                token=token,
                revision=revision,
                force_download=force_download,
            )
        return cls(variant=spec, model_path=model_path)

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 1.0,
        **kwargs: Any,
    ) -> Image.Image:
        array, _ = self.generate_array(
            prompt,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            guidance=guidance,
            **kwargs,
        )
        return Image.fromarray(np.array(array))

    def prepare_prompt(
        self,
        prompt: str,
        *,
        auto_json_caption: bool = True,
        prompt_expansion_model: str | None = None,
        width: int | None = None,
        height: int | None = None,
        warn: bool = True,
    ) -> NormalizedPrompt:
        return prepare_ideogram_prompt(
            prompt,
            auto_json_caption=auto_json_caption,
            prompt_expansion_model=prompt_expansion_model,
            width=width,
            height=height,
            warn=warn,
        )

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 0,
        steps: int = 4,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 1.0,
        **kwargs: Any,
    ) -> tuple[mx.array, dict[str, Any]]:
        validate_dimensions(width=width, height=height)
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        auto_json_value = kwargs.get("auto_json_caption", True)
        auto_json_caption = True if auto_json_value is None else bool(auto_json_value)
        prepared_prompt = self.prepare_prompt(
            prompt,
            auto_json_caption=auto_json_caption,
            prompt_expansion_model=kwargs.get("prompt_expansion_model"),
            width=width,
            height=height,
        )

        preset = get_preset(kwargs.get("sampler_preset"))
        num_steps = int(
            kwargs.get("num_steps") or (preset.num_steps if steps == 4 else steps)
        )
        if num_steps < 1:
            raise ValueError(f"steps must be >= 1, got {num_steps}")

        guidance_schedule = kwargs.get("guidance_schedule")
        if guidance_schedule is None and num_steps == preset.num_steps:
            guidance_schedule = preset.guidance_schedule
        guidance_scale = float(
            kwargs.get("guidance_scale", guidance if guidance != 1.0 else 7.0)
        )
        if guidance_schedule is not None:
            guidance_schedule = tuple(float(item) for item in guidance_schedule)
            if len(guidance_schedule) != num_steps:
                raise ValueError(
                    f"guidance_schedule must have {num_steps} items, "
                    f"got {len(guidance_schedule)}"
                )

        mu = float(kwargs.get("mu", preset.mu))
        std = float(kwargs.get("std", preset.std))
        inputs = self._build_inputs(prepared_prompt.text, height=height, width=width)
        llm_features = self._encode_text(
            inputs["text_token_ids"],
            num_image_tokens=inputs["num_image_tokens"],
        )
        if self.runtime_config.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()

        self._ensure_transformers_and_vae()
        batch_size = 1
        num_image_tokens = inputs["num_image_tokens"]
        grid_h = inputs["grid_h"]
        grid_w = inputs["grid_w"]
        num_text_tokens = inputs["num_text_tokens"]

        mx.random.seed(seed)
        z = mx.random.normal(
            (batch_size, num_image_tokens, LATENT_DIM), dtype=mx.float32
        )
        text_z_padding = mx.zeros(
            (batch_size, num_text_tokens, LATENT_DIM), dtype=mx.float32
        )

        schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
        intervals = make_step_intervals(num_steps)
        neg_position_ids = inputs["position_ids"][:, num_text_tokens:]
        neg_segment_ids = inputs["segment_ids"][:, num_text_tokens:]
        neg_indicator = inputs["indicator"][:, num_text_tokens:]

        for i in range(num_steps - 1, -1, -1):
            t_val = schedule(intervals[i + 1])
            s_val = schedule(intervals[i])
            t = mx.full((batch_size,), t_val, dtype=mx.float32)

            pos_z = mx.concatenate([text_z_padding, z], axis=1)
            pos_out = self.conditional_transformer(  # type: ignore[operator]
                llm_features=llm_features,
                x=pos_z,
                t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
            )
            pos_v = pos_out[:, num_text_tokens:]
            neg_v = self.unconditional_transformer(  # type: ignore[operator]
                llm_features=None,
                x=z,
                t=t,
                position_ids=neg_position_ids,
                segment_ids=neg_segment_ids,
                indicator=neg_indicator,
            )
            gw_i = (
                guidance_schedule[i]
                if guidance_schedule is not None
                else guidance_scale
            )
            velocity = gw_i * pos_v + (1.0 - gw_i) * neg_v
            z = z + velocity * (s_val - t_val)
            mx.eval(z)

        array = self._decode(z, grid_h=grid_h, grid_w=grid_w)
        if self.runtime_config.evict_transformers:
            self.conditional_transformer = None
            self.unconditional_transformer = None
            self.vae = None
            gc.collect()
            mx.clear_cache()
        return array, {
            "model_path": str(self.model_path),
            "sampler_preset": kwargs.get("sampler_preset") or "V4_DEFAULT_20",
            "steps": num_steps,
            "guidance": guidance_scale,
            "guidance_schedule": (
                list(guidance_schedule) if guidance_schedule is not None else None
            ),
            "mu": mu,
            "std": std,
            "prompt_tokens": int(inputs["num_text_tokens"]),
            "architecture": "single_stream_dit",
            "weight_load": "fp8_dequantized_to_bf16",
            "auto_json_caption": auto_json_caption,
            "prompt_was_wrapped": prepared_prompt.was_wrapped,
            "prompt_is_json_caption": prepared_prompt.is_json_caption,
            "prompt_is_structured_caption": prepared_prompt.is_structured_caption,
            "prompt_warnings": list(prepared_prompt.warnings),
            "revised_prompt": (
                prepared_prompt.text
                if prepared_prompt.was_wrapped or prepared_prompt.prompt_expansion_used
                else None
            ),
            "prompt_expansion_model": prepared_prompt.prompt_expansion_model,
            "prompt_expansion_used": prepared_prompt.prompt_expansion_used,
            "prompt_expansion_error": prepared_prompt.prompt_expansion_error,
        }

    def _ensure_text_encoder(self) -> None:
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder(self.model_path)

    def _ensure_transformers_and_vae(self) -> None:
        if self.conditional_transformer is None:
            self.conditional_transformer = load_transformer(
                self.model_path, subfolder="transformer"
            )
        if self.unconditional_transformer is None:
            self.unconditional_transformer = load_transformer(
                self.model_path, subfolder="unconditional_transformer"
            )
        if self.vae is None:
            self.vae = load_vae(self.model_path)

    def _tokenize(self, prompt: str) -> mx.array:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        num_text_tokens = len(token_ids)
        if num_text_tokens > MAX_TEXT_TOKENS:
            raise ValueError(
                f"prompt has {num_text_tokens} tokens, "
                f"exceeds max_text_tokens={MAX_TEXT_TOKENS}"
            )
        return mx.array(token_ids, dtype=mx.int32)

    def _build_inputs(self, prompt: str, *, height: int, width: int) -> dict[str, Any]:
        token_ids = self._tokenize(prompt)
        num_text_tokens = token_ids.shape[0]
        patch = PATCH_SIZE * AE_SCALE_FACTOR
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"height/width must be divisible by {patch}")
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w

        text_pos = mx.arange(num_text_tokens, dtype=mx.int32)
        text_pos_3d = mx.stack([text_pos, text_pos, text_pos], axis=1)

        h_idx = mx.broadcast_to(
            mx.arange(grid_h, dtype=mx.int32).reshape(-1, 1), (grid_h, grid_w)
        ).reshape(-1)
        w_idx = mx.broadcast_to(
            mx.arange(grid_w, dtype=mx.int32).reshape(1, -1), (grid_h, grid_w)
        ).reshape(-1)
        t_idx = mx.zeros_like(h_idx)
        image_pos = mx.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET
        position_ids = mx.expand_dims(
            mx.concatenate([text_pos_3d, image_pos], axis=0), axis=0
        )

        text_indicator = mx.full(
            (num_text_tokens,), LLM_TOKEN_INDICATOR, dtype=mx.int32
        )
        image_indicator = mx.full(
            (num_image_tokens,), OUTPUT_IMAGE_INDICATOR, dtype=mx.int32
        )
        indicator = mx.expand_dims(
            mx.concatenate([text_indicator, image_indicator], axis=0), axis=0
        )
        segment_ids = mx.ones((1, num_text_tokens + num_image_tokens), dtype=mx.int32)
        return {
            "text_token_ids": mx.expand_dims(token_ids, axis=0),
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "indicator": indicator,
            "num_text_tokens": int(num_text_tokens),
            "num_image_tokens": int(num_image_tokens),
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
        }

    def _encode_text(self, token_ids: mx.array, *, num_image_tokens: int) -> mx.array:
        self._ensure_text_encoder()
        attention_mask = mx.ones(token_ids.shape, dtype=mx.int32)
        _, hidden_states_list = self.text_encoder(  # type: ignore[operator]
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hidden_states_list is None:
            raise RuntimeError("Qwen3-VL hidden states were not returned")
        selected = [
            hidden_states_list[layer + 1] for layer in QWEN3_VL_ACTIVATION_LAYERS
        ]
        stacked = mx.stack(selected, axis=0)
        stacked = mx.transpose(stacked, (1, 2, 3, 0))
        batch_size, seq_len, hidden_dim, num_layers = stacked.shape
        prompt_embeds = stacked.reshape(batch_size, seq_len, hidden_dim * num_layers)
        image_padding = mx.zeros(
            (batch_size, num_image_tokens, prompt_embeds.shape[-1]),
            dtype=prompt_embeds.dtype,
        )
        return mx.concatenate([prompt_embeds, image_padding], axis=1).astype(mx.float32)

    def _decode(self, z: mx.array, *, grid_h: int, grid_w: int) -> mx.array:
        shift, scale = get_latent_norm(dtype=z.dtype)
        z = z * scale.reshape(1, 1, -1) + shift.reshape(1, 1, -1)

        patch = PATCH_SIZE
        batch_size = z.shape[0]
        ae_channels = z.shape[-1] // (patch * patch)
        latents = z.reshape(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        latents = mx.transpose(latents, (0, 5, 1, 3, 2, 4))
        latents = latents.reshape(
            batch_size, ae_channels, grid_h * patch, grid_w * patch
        )
        decoded = self.vae.decode(latents.astype(PRECISION))  # type: ignore[union-attr]
        decoded = mx.clip(decoded.astype(mx.float32), -1.0, 1.0)
        decoded = mx.round((decoded + 1.0) * 127.5).astype(mx.uint8)
        decoded = mx.transpose(decoded, (0, 2, 3, 1))
        mx.eval(decoded)
        return decoded[0]
