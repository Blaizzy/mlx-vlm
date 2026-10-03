from __future__ import annotations

import gc
import math
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from ..flux2.latent import patchify_latents
from .conditioning import format_prompt, query_attention_mask
from .config import read_config, validate_dimensions, validate_model_layout
from .scheduler import LLaDAImageScheduler
from .vq_sampling import generate_vq_tokens, vq_prompt_ids
from .weights import (
    load_lm_head,
    load_queryformer,
    load_sigvq,
    load_text_encoder,
    load_text_projection,
    load_transformer,
    load_vae,
)


class LLaDAImagePipeline:
    def __init__(
        self,
        model_path: str | Path,
        *,
        evict_text_encoder: bool = True,
        evict_transformer: bool = True,
        max_sequence_length: int = 2048,
        prompt_cache_size: int = 2,
    ):
        self.model_path = validate_model_layout(model_path)
        if max_sequence_length < 1 or prompt_cache_size < 0:
            raise ValueError("Invalid prompt length or cache size")
        self.max_sequence_length = max_sequence_length
        self.prompt_cache_size = prompt_cache_size
        self.evict_text_encoder = evict_text_encoder
        self.evict_transformer = evict_transformer
        self.scheduler = LLaDAImageScheduler(
            read_config(self.model_path / "scheduler" / "scheduler_config.json")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
        )
        self.text_encoder = None
        self.queryformer = None
        self.text_projection = None
        self.transformer = None
        self.vae = None
        self.prompt_cache: OrderedDict[str | None, mx.array] = OrderedDict()

    def _release(self, *components: str):
        for name in components:
            setattr(self, name, None)
        gc.collect()
        mx.clear_cache()

    def tokenize(self, prompt: str | None) -> list[int]:
        return self.tokenizer.encode(
            format_prompt(prompt),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_sequence_length,
        )

    def _encode_prompt(self, prompt: str | None) -> mx.array:
        if prompt in self.prompt_cache:
            self.prompt_cache.move_to_end(prompt)
            return self.prompt_cache[prompt]
        if self.text_encoder is None:
            # A new prompt may follow an earlier generation on the same model.
            if self.evict_transformer:
                self._release("transformer", "vae")
            self.text_encoder = load_text_encoder(self.model_path)
            self.queryformer = load_queryformer(self.model_path)
            self.text_projection = load_text_projection(self.model_path)
        ids = mx.array([self.tokenize(prompt)], dtype=mx.int32)
        embeddings = self.text_encoder.word_embeddings(ids)
        queries = self.queryformer(embeddings, mx.ones(ids.shape, dtype=mx.bool_))
        combined = mx.concatenate([embeddings, queries], axis=1)
        # Encode each prompt without padding. Its positions are contiguous, so
        # the backbone's ordinary RoPE offsets match the reference position_ids.
        hidden = self.text_encoder(
            ids,
            inputs_embeds=combined,
            mask=query_attention_mask(ids.shape[1], queries.shape[1]),
        )
        output = self.text_projection(hidden)
        mx.eval(output)
        if self.prompt_cache_size:
            self.prompt_cache[prompt] = output
            while len(self.prompt_cache) > self.prompt_cache_size:
                self.prompt_cache.popitem(last=False)
        return output

    def _generate_semantic_features(self, prompt, height, width):
        if self.text_encoder is None:
            if self.evict_transformer:
                self._release("transformer", "vae")
            self.text_encoder = load_text_encoder(self.model_path)
        ids, unconditional, count = vq_prompt_ids(self.tokenizer, prompt, height, width)
        head = load_lm_head(self.model_path)
        config = read_config(self.model_path / "sigvq" / "config.json")
        token_ids = generate_vq_tokens(
            self.text_encoder,
            head,
            ids,
            unconditional,
            count,
            codebook_size=config["codebook_size"],
        )
        mx.eval(token_ids)
        del head
        if self.evict_text_encoder:
            self._release("text_encoder", "queryformer", "text_projection")
        sigvq = load_sigvq(self.model_path, include_encoder=False)
        features = sigvq(token_ids=token_ids)
        mx.eval(features)
        return features

    def _encode_source_image(self, image: Image.Image, height: int, width: int):
        pixels = np.array(
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        pixels = mx.array(pixels)[None] / 127.5 - 1
        # Half-size bilinear interpolation with align_corners=False averages 2x2 pixels.
        half = pixels.reshape(1, height // 2, 2, width // 2, 2, 3).mean(axis=(2, 4))
        sigvq = load_sigvq(self.model_path)
        semantic = sigvq(pixels=half.astype(mx.bfloat16))
        mx.eval(semantic)
        del sigvq
        mx.clear_cache()
        if self.vae is None or self.vae.encoder is None:
            self.vae = load_vae(self.model_path, include_encoder=True)
        source = patchify_latents(
            self.vae.encode(pixels.transpose(0, 3, 1, 2).astype(mx.bfloat16))
        )
        mean = self.vae.bn.running_mean.reshape(1, -1, 1, 1).astype(source.dtype)
        std = mx.sqrt(
            self.vae.bn.running_var.reshape(1, -1, 1, 1) + self.vae.bn.eps
        ).astype(source.dtype)
        source = (source - mean) / std
        mx.eval(source)
        return source, semantic

    def edit_array(
        self, prompt: str, image_path: str | Path, *, width=None, height=None, **kwargs
    ):
        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        if width is None and height is None:
            scale = min(1.0, math.sqrt(1024 * 1024 / (image.width * image.height)))
            width = max(32, int(image.width * scale) // 32 * 32)
            height = max(32, int(image.height * scale) // 32 * 32)
        elif width is None:
            width = max(32, round(image.width * height / image.height / 32) * 32)
        elif height is None:
            height = max(32, round(image.height * width / image.width / 32) * 32)
        return self.generate_array(
            prompt,
            width=width,
            height=height,
            generation_mode="editing",
            image=image,
            **kwargs,
        )

    def generate_array(
        self,
        prompt: str,
        *,
        seed: int = 0,
        steps: int | None = None,
        width: int = 1024,
        height: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        stochastic_sampling: bool | None = None,
        generation_mode: str = "text",
        image: Image.Image | None = None,
    ) -> mx.array:
        validate_dimensions(width, height)
        if generation_mode not in {"text", "vq", "editing"}:
            raise ValueError("generation_mode must be text, vq, or editing")
        if (image is not None) != (generation_mode == "editing"):
            raise ValueError("An input image is required only for editing mode")
        if generation_mode == "editing" and (width % 32 or height % 32):
            raise ValueError("Editing width and height must be multiples of 32")
        steps = self.scheduler.default_steps if steps is None else steps
        guidance = self.scheduler.default_guidance if guidance is None else guidance
        if steps < 1 or not math.isfinite(guidance) or guidance < 0:
            raise ValueError(
                "steps must be positive and guidance finite and nonnegative"
            )
        caption = self._encode_prompt(prompt)
        negative = self._encode_prompt(negative_prompt) if guidance > 1.0 else None
        semantic, source = None, None
        if generation_mode == "vq":
            semantic = self._generate_semantic_features(prompt, height, width)
        if self.evict_text_encoder:
            self._release("text_encoder", "queryformer", "text_projection")
        if generation_mode == "editing":
            source, semantic = self._encode_source_image(image, height, width)
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path)
        dtype = self.transformer.x_embedder.weight.dtype
        caption = caption.astype(dtype)
        if negative is not None:
            negative = negative.astype(dtype)
        if semantic is not None:
            semantic = semantic.astype(dtype)
        if source is not None:
            source = source.astype(dtype)
        # Explicit seeding after module construction also makes cache hits and
        # component reloading produce the same diffusion noise.
        mx.random.seed(seed)
        latents = mx.random.normal((1, 128, height // 16, width // 16))
        latents = latents.astype(dtype).astype(mx.float32)
        sigmas = self.scheduler.sigmas(steps)
        scheduler = self.scheduler
        if stochastic_sampling is not None:
            scheduler = LLaDAImageScheduler(
                {
                    "use_uniform_sigmas": self.scheduler.use_uniform_sigmas,
                    "shift": self.scheduler.shift,
                    "stochastic_sampling": stochastic_sampling,
                }
            )
        for index in range(steps):
            time = sigmas[index : index + 1].astype(dtype)
            model_input = latents.astype(dtype)
            prediction = -self.transformer(
                model_input, time, caption, semantic=semantic, source_latents=source
            ).astype(mx.float32)
            if negative is not None:
                unconditional = -self.transformer(
                    model_input,
                    time,
                    negative,
                    semantic=semantic[:, :0] if semantic is not None else None,
                    source_latents=source,
                ).astype(mx.float32)
                prediction = unconditional + guidance * (prediction - unconditional)
            latents = scheduler.step(
                prediction, latents, sigmas[index], sigmas[index + 1]
            )
            mx.eval(latents)
        # Release the last graph references as well as the module before decoding.
        del prediction, model_input
        if negative is not None:
            del unconditional
        if self.evict_transformer:
            self._release("transformer")
        if self.vae is None:
            self.vae = load_vae(self.model_path)
        decoded = self.vae.decode_packed_latents(latents.astype(mx.bfloat16))
        image = mx.clip(decoded.astype(mx.float32) / 2 + 0.5, 0, 1)
        image = mx.round(image[0].transpose(1, 2, 0) * 255).astype(mx.uint8)
        mx.eval(image)
        return image
