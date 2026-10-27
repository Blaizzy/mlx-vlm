from __future__ import annotations

import gc
import math
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

from .scheduler import MOSSImageScheduler
from .transformer import validate_skip_layers
from .weights import load_text_encoder, load_transformer, load_vae, read_config


class MOSSImagePipeline:
    def __init__(self, model_path: str | Path, *, evict_text_encoder=True):
        self.model_path = Path(model_path).expanduser()
        self.evict_text_encoder = evict_text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path / "tokenizer"), local_files_only=True
        )
        self.text_encoder = self.transformer = self.vae = None
        self.transformer_config = read_config(
            self.model_path / "transformer/config.json"
        )
        self.scheduler_config = read_config(
            self.model_path / "scheduler/scheduler_config.json"
        )

    def _format_prompt(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def count_prompt_tokens(self, prompt):
        return len(self.tokenizer(self._format_prompt(prompt))["input_ids"])

    def _encode_prompt(self, prompt, max_sequence_length):
        # Individual, unpadded forwards avoid padding effects in Qwen's GDN.
        tokens = self.tokenizer(
            self._format_prompt(prompt),
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        # The shared Qwen backbone accepts explicit positions for a forward
        # without a KV cache (the diffusion encoder never generates tokens).
        positions = mx.broadcast_to(
            mx.arange(input_ids.shape[1])[None, None], (3, 1, input_ids.shape[1])
        )
        hidden = self.text_encoder(input_ids, position_ids=positions)
        mx.eval(hidden)
        return hidden

    def generate_array(
        self,
        prompt,
        *,
        seed=42,
        steps=50,
        width=1024,
        height=1024,
        guidance=3.0,
        negative_prompt="",
        num_images=1,
        max_sequence_length=4096,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale=0.0,
        slg_start=0.0,
        slg_stop=1.0,
    ):
        for name, value in (("width", width), ("height", height)):
            if type(value) is not int or value <= 0 or value % 16:
                raise ValueError(f"MOSS-Image {name} must be a positive multiple of 16")
        if type(num_images) is not int or num_images < 1:
            raise ValueError("num_images must be a positive integer")
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError("slg_scale must be finite and nonnegative")
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError("Require 0 <= slg_start <= slg_stop <= 1")
        if type(max_sequence_length) is not int or not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096")
        if not isinstance(prompt, str) or not isinstance(negative_prompt, str):
            raise TypeError("prompt and negative_prompt must be strings")
        depth = self.transformer_config["num_layers"]
        uncond_skip_layers = validate_skip_layers(
            uncond_skip_layers, depth, "uncond_skip_layers"
        )
        slg_layers = validate_skip_layers(slg_layers, depth, "slg_layers")
        scheduler = MOSSImageScheduler(
            **{
                key: self.scheduler_config[key]
                for key in ("num_train_timesteps", "shift", "training_shift")
                if key in self.scheduler_config
            }
        )
        scheduler.set_timesteps(steps)

        if self.evict_text_encoder:
            self.transformer = self.vae = None
            gc.collect()
            mx.clear_cache()
        if self.text_encoder is None:
            self.text_encoder = load_text_encoder(self.model_path)
        positive = self._encode_prompt(prompt, max_sequence_length)
        negative = (
            self._encode_prompt(negative_prompt, max_sequence_length)
            if guidance != 1
            else None
        )
        if self.evict_text_encoder:
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path)

        # Preserve fp32 noise, CFG arithmetic, and Euler updates with bf16 DiT.
        latents = mx.random.normal(
            (
                num_images,
                self.transformer_config["latent_channels"],
                height // 16,
                width // 16,
            ),
            dtype=mx.float32,
            key=mx.random.key(seed),
        )
        positive = mx.repeat(positive, num_images, axis=0)
        if negative is not None:
            negative = mx.repeat(negative, num_images, axis=0)
        num_steps = len(scheduler.timesteps)
        for i in range(num_steps):
            timestep = scheduler.timesteps[i]
            skip_active = slg_start <= i / num_steps < slg_stop
            conditional = self.transformer(latents, timestep, positive).astype(
                mx.float32
            )
            prediction = conditional
            if negative is not None:
                uncond = self.transformer(
                    latents,
                    timestep,
                    negative,
                    skip_layers=uncond_skip_layers if skip_active else None,
                ).astype(mx.float32)
                prediction = uncond + guidance * (conditional - uncond)
            if slg_layers and slg_scale > 0 and skip_active:
                skipped = self.transformer(
                    latents, timestep, positive, skip_layers=slg_layers
                ).astype(mx.float32)
                if negative is None:
                    prediction = skipped
                prediction = prediction + slg_scale * (conditional - skipped)
            latents = scheduler.step(prediction, i, latents)
            mx.eval(latents)
        if self.vae is None:
            self.vae = load_vae(self.model_path)
        decoded = self.vae.decode_packed_latents(latents)
        images = mx.round(
            mx.clip(decoded.transpose(0, 2, 3, 1) / 2 + 0.5, 0, 1) * 255
        ).astype(mx.uint8)
        mx.eval(images)
        return images[0] if num_images == 1 else images
