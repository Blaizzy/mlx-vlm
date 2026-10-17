from __future__ import annotations

import gc
import json
import math
from itertools import pairwise
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps

from ..flux2.latent import prepare_grid_ids
from ..flux2.prompt import prepare_text_ids
from . import bbox
from .config import Flux3Config, validate_dimensions
from .download import download_model, validate_model_layout
from .text_encoder import Flux3Tokenizer
from .upsampler import DEFAULT_UPSAMPLER, Flux3PromptUpsampler
from .weights import load_text_encoder, load_transformer, load_vae


def timesteps(steps: int, alpha: float = 6.93) -> mx.array:
    if steps < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("steps and alpha must be positive")
    t = mx.linspace(1, 0, steps + 1, dtype=mx.float32)
    return alpha * t / (1 + (alpha - 1) * t)


def image_ids(height, width, *, reference=None, scale=0.5):
    grid = prepare_grid_ids(mx.zeros((1, 1, height // 16, width // 16)), t_coord=0)
    factors = mx.array([1, scale, scale, 1], dtype=mx.float32)
    ids = grid.astype(mx.float32) * factors
    if reference is not None:
        ids = ids + mx.array([0, 0, 0, -1 - reference], dtype=mx.float32)
    return ids


def prepare_reference(image, *, multi=False):
    """Preserve reference aspect ratios and the upstream single/multi size budgets."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if multi:
        scale = min(1, math.sqrt(1024 * 1024 / (w * h)))
        w, h = max(16, int(w * scale)), max(16, int(h * scale))
        if (w, h) != image.size:
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        left, top = (w % 16) // 2, (h % 16) // 2
        return image.crop((left, top, left + w // 16 * 16, top + h // 16 * 16)), None
    scale = min(1, math.sqrt(4 * 1024 * 1024 / (w * h)), 4096 / max(w, h))
    while True:
        fit = (max(1, round(w * scale)), max(1, round(h * scale)))
        canvas = tuple(min(4096, max(64, math.ceil(v / 16) * 16)) for v in fit)
        if canvas[0] * canvas[1] <= 4 * 1024 * 1024:
            break
        scale *= 0.995
    if image.size != fit:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    left, top = (canvas[0] - fit[0]) // 2, (canvas[1] - fit[1]) // 2
    padding = (
        (top, canvas[1] - fit[1] - top),
        (left, canvas[0] - fit[0] - left),
        (0, 0),
    )
    mode = "reflect" if min(fit) > 16 else "symmetric"
    crop = (left, top, left + fit[0], top + fit[1]) if canvas != fit else None
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode)), crop


class Flux3Image:
    def __init__(
        self,
        model_path,
        *,
        evict_text_encoder=True,
        evict_transformer=False,
        max_sequence_length=4096,
        upsample_prompt=True,
        prompt_expansion_model=DEFAULT_UPSAMPLER,
        evict_upsampler=True,
    ):
        self.model_path = validate_model_layout(model_path)
        self.config = Flux3Config.from_path(self.model_path)
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.max_sequence_length = min(max_sequence_length, self.config.max_tokens)
        self.evict_text_encoder = evict_text_encoder
        self.evict_transformer = evict_transformer
        self.upsample_prompt = upsample_prompt
        self.prompt_expansion_model = str(prompt_expansion_model)
        self.evict_upsampler = evict_upsampler
        self.upsampler = None
        self.upsampled_prompts = {}
        self.last_prompt_metadata = {}
        self.tokenizer = Flux3Tokenizer(
            self.model_path / "text_encoder",
            max_length=self.config.max_tokens,
            pad_multiple=self.config.pad_multiple,
        )
        self.text_encoder = None
        self.transformer = None
        self.vae = None
        self.prompt_cache = {}

    @classmethod
    def from_pretrained(
        cls,
        model="black-forest-labs/flux-3-image-klein",
        *,
        model_path=None,
        download=True,
        token=None,
        revision=None,
        force_download=False,
        **kwargs,
    ):
        root = Path(model).expanduser()
        if model_path is None and root.is_dir():
            model_path = root
        if model_path is None:
            if not download:
                raise FileNotFoundError(f"No local model_path provided for {model}")
            model_path = download_model(
                model, token=token, revision=revision, force_download=force_download
            )
        return cls(model_path, **kwargs)

    def _encode_prompt(self, prompt, max_sequence_length=None):
        cap = (
            self.max_sequence_length
            if max_sequence_length is None
            else max_sequence_length
        )
        if cap < 1:
            raise ValueError("max_sequence_length must be positive")
        key = (prompt, cap)
        if key not in self.prompt_cache:
            if self.text_encoder is None:
                self.text_encoder = load_text_encoder(
                    self.model_path, self.config.text_layers
                )
            tokens, mask = self.tokenizer.tokenize(prompt, max_length=cap)
            embeds = self.text_encoder.get_prompt_embeds(
                tokens, mask, self.config.text_layers
            )
            ids = prepare_text_ids(embeds)
            mx.eval(embeds, ids)
            # Bound retained prompts for long-running servers.
            if len(self.prompt_cache) >= 4:
                self.prompt_cache.pop(next(iter(self.prompt_cache)))
            self.prompt_cache[key] = (embeds, ids)
            if self.evict_text_encoder:
                self.text_encoder = None
                gc.collect()
                mx.clear_cache()
        return self.prompt_cache[key]

    def _prepare_prompt(
        self, prompt, images, *, upsample_prompt, prompt_expansion_model, boxes
    ):
        enabled = self.upsample_prompt if upsample_prompt is None else upsample_prompt
        model_id = str(prompt_expansion_model or self.prompt_expansion_model)
        self.last_prompt_metadata = {"input_prompt": prompt, "upsampled": False}
        if not enabled or (not boxes and bbox.is_finished(prompt)):
            return (
                f"{prompt.rstrip()} {json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)}"
                if boxes
                else prompt
            )
        key = (prompt, model_id, json.dumps(boxes, sort_keys=True))
        caption = self.upsampled_prompts.get(key) if not images else None
        if caption is None:
            if self.evict_upsampler:
                # Keep the 9B VLM and diffusion components in separate stages,
                # including on subsequent requests using the same pipeline.
                self.text_encoder = self.transformer = self.vae = None
                gc.collect()
                mx.clear_cache()
            if self.upsampler is None or self.upsampler.model_id != model_id:
                if self.upsampler is not None:
                    self.upsampler.unload()
                self.upsampler = Flux3PromptUpsampler(
                    model_id, evict_model=self.evict_upsampler
                )
            caption = self.upsampler(prompt, images, boxes=boxes)
            if not images:
                if len(self.upsampled_prompts) >= 4:
                    self.upsampled_prompts.pop(next(iter(self.upsampled_prompts)))
                self.upsampled_prompts[key] = caption
        self.last_prompt_metadata.update(
            upsampled=True, revised_prompt=caption, prompt_expansion_model=model_id
        )
        return caption

    def generate(self, prompt, **kwargs):
        return Image.fromarray(np.array(self.generate_array(prompt, **kwargs)))

    def edit(self, prompt, image_paths, **kwargs):
        return Image.fromarray(np.array(self.edit_array(prompt, image_paths, **kwargs)))

    def generate_array(
        self,
        prompt,
        *,
        seed=42,
        steps=None,
        width=1024,
        height=1024,
        guidance=None,
        max_sequence_length=None,
        num_images=1,
        upsample_prompt=None,
        prompt_expansion_model=None,
        boxes=None,
    ):
        return self._run(
            prompt,
            (),
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            guidance=guidance,
            max_sequence_length=max_sequence_length,
            num_images=num_images,
            upsample_prompt=upsample_prompt,
            prompt_expansion_model=prompt_expansion_model,
            boxes=boxes,
        )

    def edit_array(
        self,
        prompt,
        image_paths,
        *,
        seed=42,
        steps=None,
        width=None,
        height=None,
        guidance=None,
        max_sequence_length=None,
        upsample_prompt=None,
        prompt_expansion_model=None,
        boxes=None,
    ):
        if not image_paths or len(image_paths) > 10:
            raise ValueError("Flux3 editing requires between 1 and 10 reference images")
        return self._run(
            prompt,
            image_paths,
            seed=seed,
            steps=steps,
            width=width,
            height=height,
            guidance=guidance,
            max_sequence_length=max_sequence_length,
            upsample_prompt=upsample_prompt,
            prompt_expansion_model=prompt_expansion_model,
            boxes=boxes,
        )

    def _run(
        self,
        prompt,
        image_paths,
        *,
        seed,
        steps,
        width,
        height,
        guidance,
        max_sequence_length,
        num_images=1,
        upsample_prompt=None,
        prompt_expansion_model=None,
        boxes=None,
    ):
        steps = self.config.default_steps if steps is None else steps
        guidance = self.config.default_guidance if guidance is None else guidance
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if num_images < 1 or steps < 1:
            raise ValueError("num_images and steps must be positive")
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and non-negative")
        if self.config.flow_map and guidance != 1:
            raise ValueError(
                "The distilled Flux3 flow map runs without CFG; use guidance=1"
            )
        prepared, originals = [], []
        for path in image_paths:
            with Image.open(Path(path).expanduser()) as image:
                originals.append(image.copy())
                prepared.append(prepare_reference(image, multi=len(image_paths) > 1))
        references = [image for image, _ in prepared]
        crop = (
            prepared[0][1]
            if len(prepared) == 1 and width is None and height is None
            else None
        )
        if (width is None) != (height is None):
            raise ValueError("Both width and height are required when either is set")
        if width is None:
            width, height = references[0].size if references else (1024, 1024)
        validate_dimensions(width, height)
        prompt = self._prepare_prompt(
            prompt,
            originals,
            upsample_prompt=upsample_prompt,
            prompt_expansion_model=prompt_expansion_model,
            boxes=boxes,
        )
        embeds, text_ids = self._encode_prompt(prompt, max_sequence_length)
        negative = (
            self._encode_prompt("", max_sequence_length)
            if not self.config.flow_map and guidance != 1
            else None
        )
        if self.transformer is None:
            self.transformer = load_transformer(self.model_path, self.config)
        if self.vae is None:
            self.vae = load_vae(self.model_path)
        data = {"text": embeds}
        ids = {"text": text_ids}
        if references:
            ref_tokens, ref_ids = [], []
            for index, reference in enumerate(references):
                pixels = mx.array(np.asarray(reference).astype(np.float32) / 127.5 - 1)[
                    None
                ]
                tokens = self.vae.encode(pixels).astype(mx.bfloat16)
                ref_tokens.append(tokens)
                ref_ids.append(
                    image_ids(
                        reference.height,
                        reference.width,
                        reference=index,
                        scale=self.config.image_rope_scale,
                    )
                )
            data["image_cond"] = mx.concatenate(ref_tokens, axis=1)
            ids["image_cond"] = mx.concatenate(ref_ids, axis=1)
            mx.eval(data["image_cond"])
        ids["image"] = image_ids(height, width, scale=self.config.image_rope_scale)
        for mode, value in data.items():
            data[mode] = mx.broadcast_to(value, (num_images, *value.shape[1:]))
        for mode, value in ids.items():
            ids[mode] = mx.broadcast_to(value, (num_images, *value.shape[1:]))
        # Match the reference: BF16 initial noise, FP32 state for Euler updates.
        latents = (
            mx.random.normal(
                (
                    num_images,
                    height // 16 * (width // 16),
                    self.config.in_channels["image"],
                ),
                key=mx.random.key(seed),
            )
            .astype(mx.bfloat16)
            .astype(mx.float32)
        )
        schedule = timesteps(steps, self.config.alpha).tolist()
        for t, target in pairwise(schedule):
            data["image"] = latents
            args = {"target_timestep": target} if self.config.flow_map else {}
            velocity = self.transformer(data, ids, t, **args)
            if negative is not None:
                negative_data = {
                    **data,
                    "text": mx.broadcast_to(
                        negative[0], (num_images, *negative[0].shape[1:])
                    ),
                }
                negative_ids = {
                    **ids,
                    "text": mx.broadcast_to(
                        negative[1], (num_images, *negative[1].shape[1:])
                    ),
                }
                uncond = self.transformer(negative_data, negative_ids, t)
                velocity = uncond + guidance * (velocity - uncond)
            latents = latents + (target - t) * velocity.astype(mx.float32)
            mx.eval(latents)
        decoded = self.vae.decode(latents, height=height, width=width)
        images = ((mx.clip(decoded, -1, 1) + 1) * 127.5).astype(mx.uint8)
        if crop is not None:
            left, top, right, bottom = crop
            images = images[:, top:bottom, left:right]
        mx.eval(images)
        if self.evict_transformer:
            self.transformer = self.vae = None
            gc.collect()
            mx.clear_cache()
        return images[0] if num_images == 1 else images
