from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mlx.core as mx

from .audio_vae import MiniMaxH3AudioVAE
from .conditioner import MiniMaxH3Conditioner, MiniMaxH3ConditioningOutput
from .constants import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
)
from .packing import (
    MiniMaxH3PackedSequence,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .processing import (
    decode_audio,
    decode_image,
    decode_video,
    decode_video_soundtrack,
    normalize_visual_vae_pixels,
    prepare_keyframe_image,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    resample_reference_frames,
)
from .references import (
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    build_ref2va_packed_sequence,
    resolve_reference_image_size,
    trim_reference_num_frames,
    validate_references,
)
from .scheduler import MiniMaxH3Scheduler
from .transformer import MiniMaxH3Transformer
from .visual_vae import MiniMaxH3DiagonalGaussianDistribution, MiniMaxH3VideoVAE


@dataclass(slots=True)
class MiniMaxH3GenerationRequest:
    prompt: str
    image: object | None = None
    last_image: object | None = None
    references: list[MiniMaxH3Reference] | None = None
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    num_inference_steps: int = 30
    seed: int = 0
    output_type: Literal["array", "latent"] = "array"
    latents: mx.array | None = None
    audio_latents: mx.array | None = None


@dataclass(frozen=True, slots=True)
class MiniMaxH3PipelineOutput:
    video: mx.array
    audio: mx.array
    sampling_rate: int
    fps: int
    metadata: dict[str, object] = field(default_factory=dict)


class _RandomStream:
    def __init__(self, seed: int) -> None:
        self.key = mx.random.key(seed)

    def normal(self, shape: tuple[int, ...], dtype=mx.float32) -> mx.array:
        self.key, key = mx.random.split(self.key)
        return mx.random.normal(shape, dtype=dtype, key=key)


class MiniMaxH3Pipeline:
    """Joint MiniMax-H3 audio/video generation for one selected task partition."""

    def __init__(
        self,
        *,
        transformer: MiniMaxH3Transformer,
        conditioner: MiniMaxH3Conditioner,
        video_vae: MiniMaxH3VideoVAE,
        audio_vae: MiniMaxH3AudioVAE,
        partition: Literal["fl2va", "ref2va"] = "fl2va",
        scheduler: MiniMaxH3Scheduler | None = None,
        audio_scheduler: MiniMaxH3Scheduler | None = None,
    ) -> None:
        if partition not in ("fl2va", "ref2va"):
            raise ValueError(f"partition must be fl2va or ref2va, got {partition!r}")
        self.transformer = transformer
        self.conditioner = conditioner
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.partition = partition
        self.scheduler = scheduler or MiniMaxH3Scheduler(shift=12.0)
        self.audio_scheduler = audio_scheduler or MiniMaxH3Scheduler(shift=3.0)

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.transformer.config.patch_size

    def _resolve_geometry(
        self,
        height: int,
        width: int,
        num_frames: int,
    ) -> tuple[int, int, int, int]:
        ratio = self.video_vae.config.spatial_compression_ratio
        if height % ratio or width % ratio:
            raise ValueError(
                f"canvas {(height, width)} must be divisible by VAE ratio {ratio}"
            )
        return (
            video_latent_num_frames(num_frames),
            height // ratio,
            width // ratio,
            audio_latent_num_frames(num_frames),
        )

    @staticmethod
    def _validate_duration(num_frames: int) -> int:
        num_frames = align_num_frames(num_frames)
        duration = num_frames / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"aligned duration must be {MINIMAX_H3_MIN_DURATION:g} to "
                f"{MINIMAX_H3_MAX_DURATION:g} seconds, got {duration:g}"
            )
        return num_frames

    @staticmethod
    def _validate_canvas(height: int, width: int) -> None:
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError(
                f"height and width must be positive multiples of 32, got {height}x{width}"
            )

    def _encode_visual_condition(self, pixels: mx.array, *, video: bool) -> mx.array:
        values = normalize_visual_vae_pixels(pixels)
        moments = (
            self.video_vae._encode(values)
            if video
            else self.video_vae._encode_clip(values)
        )
        latents = MiniMaxH3DiagonalGaussianDistribution(moments).sample(
            mx.random.key(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        )
        # This deliberate FP16 round trip is part of the released recipe.
        return latents.astype(mx.float16).astype(mx.float32)

    def _normalize_video_latents(self, latents: mx.array) -> mx.array:
        mean = mx.array(self.video_vae.config.latents_mean, mx.float32).reshape(
            1, -1, 1, 1, 1
        )
        std = mx.array(self.video_vae.config.latents_std, mx.float32).reshape(
            1, -1, 1, 1, 1
        )
        return (latents - mean) / std

    def _normalize_audio_latents(self, latents: mx.array) -> mx.array:
        mean = mx.array(self.audio_vae.config.latents_mean, mx.float32).reshape(
            1, 1, -1
        )
        std = mx.array(self.audio_vae.config.latents_std, mx.float32).reshape(1, 1, -1)
        return (latents - mean) / std

    def _prepare_fl2va(
        self,
        request: MiniMaxH3GenerationRequest,
        random: _RandomStream,
    ) -> tuple[
        MiniMaxH3ConditioningOutput,
        MiniMaxH3PackedSequence,
        mx.array | None,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ]:
        raw_keyframes = [
            value for value in (request.image, request.last_image) if value is not None
        ]
        decoded = [decode_image(value) for value in raw_keyframes]
        anchors = tuple(
            anchor
            for anchor, value in (
                ("first", request.image),
                ("last", request.last_image),
            )
            if value is not None
        )
        if request.height is None or request.width is None:
            if request.height is not None or request.width is not None:
                raise ValueError("height and width must be passed together")
            if decoded:
                height, width = resolve_canvas_size(
                    decoded[0].shape[1], decoded[0].shape[0]
                )
            else:
                height, width = resolve_canvas_size(16, 9)
        else:
            height, width = request.height, request.width
        self._validate_canvas(height, width)
        num_frames = self._validate_duration(
            124 if request.num_frames is None else request.num_frames
        )
        geometry = self._resolve_geometry(height, width, num_frames)
        keyframes = [
            prepare_keyframe_image(
                image,
                height,
                width,
                # The pinned pipeline stretches the first item in packed
                # keyframe order, including a last-only request.
                stretch=index == 0,
            )
            for index, image in enumerate(decoded)
        ]
        conditioning = self.conditioner.encode_fl2va(request.prompt, keyframes)
        condition_rows = []
        for keyframe in keyframes:
            latents = self._normalize_video_latents(
                self._encode_visual_condition(keyframe, video=False)
            )
            condition_rows.append(patchify_video_latents(latents, self.patch_size))
        condition_latents = mx.concatenate(condition_rows) if condition_rows else None
        if condition_latents is not None:
            noise_rows = []
            for _ in keyframes:
                noise = random.normal(
                    (
                        1,
                        self.video_vae.config.latent_channels,
                        1,
                        geometry[1],
                        geometry[2],
                    )
                )
                noise_rows.append(patchify_video_latents(noise, self.patch_size))
            condition_latents = self.scheduler.scale_noise(
                condition_latents,
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                mx.concatenate(noise_rows),
            )
        layout = build_packed_sequence(
            conditioning.token_tags,
            *geometry,
            self.patch_size,
            keyframe_anchors=anchors,
        )
        return (
            conditioning,
            layout,
            condition_latents,
            num_frames,
            height,
            width,
            *geometry,
        )

    def _prepare_references(
        self,
        references: list[MiniMaxH3Reference],
        num_frames: int | None,
    ) -> tuple[list[MiniMaxH3PreparedReference], int]:
        validate_references(references)
        decoded_audio: dict[int, tuple[mx.array, int]] = {}
        for index, reference in enumerate(references):
            if (
                reference.kind == "video"
                and reference.audio is None
                and isinstance(reference.video, (str, Path))
            ):
                soundtrack = decode_video_soundtrack(reference.video)
                if soundtrack is not None:
                    decoded_audio[index] = soundtrack
        if num_frames is None:
            audio_bearing = [
                index
                for index, ref in enumerate(references)
                if ref.has_audio or index in decoded_audio
            ]
            if len(audio_bearing) != 1:
                raise ValueError(
                    "num_frames can be inferred only from exactly one audio-bearing reference"
                )
            reference_index = audio_bearing[0]
            reference = references[reference_index]
            waveform, sample_rate = decoded_audio.get(reference_index, (None, None))
            if waveform is None:
                waveform, decoded_rate = decode_audio(reference.audio)
                sample_rate = (
                    reference.sample_rate
                    or decoded_rate
                    or self.audio_vae.config.sampling_rate
                )
                decoded_audio[reference_index] = (waveform, sample_rate)
            duration = waveform.shape[-1] / sample_rate
            if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
                raise ValueError(
                    f"audio reference duration must be {MINIMAX_H3_MIN_DURATION:g} "
                    f"to {MINIMAX_H3_MAX_DURATION:g} seconds, got {duration:g}"
                )
            num_frames = round(duration * MINIMAX_H3_FPS)
        num_frames = self._validate_duration(num_frames)

        prepared = []
        for reference_index, reference in enumerate(references):
            item = MiniMaxH3PreparedReference(
                kind=reference.kind,
                has_audio=reference.has_audio or reference_index in decoded_audio,
            )
            if reference.kind == "image":
                image = decode_image(reference.image)
                height, width = resolve_reference_image_size(
                    image.shape[1], image.shape[0]
                )
                item.image = prepare_reference_image(image, height, width)
            elif reference.kind == "video":
                frames, decoded_fps = decode_video(reference.video)
                frames = resample_reference_frames(
                    frames,
                    float(reference.fps or decoded_fps or MINIMAX_H3_FPS),
                )
                item.frames = prepare_reference_frames(frames, num_frames)
            if item.has_audio:
                waveform, sample_rate = decoded_audio.get(
                    reference_index,
                    (None, None),
                )
                if waveform is None:
                    waveform, decoded_rate = decode_audio(reference.audio)
                    sample_rate = (
                        reference.sample_rate
                        or decoded_rate
                        or self.audio_vae.config.sampling_rate
                    )
                item.waveform = prepare_reference_waveform(
                    waveform,
                    sample_rate,
                    self.audio_vae.config.sampling_rate,
                    num_frames / MINIMAX_H3_FPS,
                )
            prepared.append(item)
        return prepared, num_frames

    def _encode_references(
        self,
        references: list[MiniMaxH3PreparedReference],
        random: _RandomStream,
    ) -> tuple[mx.array | None, mx.array | None]:
        video_rows = []
        audio_rows = []
        for reference in references:
            if reference.kind != "audio":
                pixels = (
                    reference.image
                    if reference.kind == "image"
                    else reference.frames[
                        : trim_reference_num_frames(reference.frames.shape[0])
                    ]
                )
                latents = self._normalize_video_latents(
                    self._encode_visual_condition(
                        pixels, video=reference.kind == "video"
                    )
                )
                reference.num_latent_frames = latents.shape[2]
                reference.latent_height = latents.shape[3]
                reference.latent_width = latents.shape[4]
                video_rows.append(patchify_video_latents(latents, self.patch_size))
            if reference.has_audio:
                latents = self.audio_vae.encode(reference.waveform[:, None]).mode()
                latents = latents.astype(mx.float32).transpose(0, 2, 1)
                reference.num_audio_latents = latents.shape[1]
                audio_rows.append(
                    self._normalize_audio_latents(latents).reshape(
                        -1, self.audio_vae.config.latent_channels
                    )
                )
        condition_video = mx.concatenate(video_rows) if video_rows else None
        condition_audio = mx.concatenate(audio_rows) if audio_rows else None
        if condition_video is not None:
            noise_rows = []
            for reference in references:
                if reference.kind != "audio":
                    noise = random.normal(
                        (
                            1,
                            self.video_vae.config.latent_channels,
                            reference.num_latent_frames,
                            reference.latent_height,
                            reference.latent_width,
                        )
                    )
                    noise_rows.append(patchify_video_latents(noise, self.patch_size))
            condition_video = self.scheduler.scale_noise(
                condition_video,
                MINIMAX_H3_KEYFRAME_NOISE_AUG,
                mx.concatenate(noise_rows),
            )
        return condition_video, condition_audio

    def _prepare_ref2va(
        self,
        request: MiniMaxH3GenerationRequest,
        random: _RandomStream,
    ) -> tuple[
        MiniMaxH3ConditioningOutput,
        MiniMaxH3PackedSequence,
        mx.array | None,
        mx.array | None,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
    ]:
        if request.image is not None or request.last_image is not None:
            raise ValueError("Ref2VA uses references, not image or last_image")
        references, num_frames = self._prepare_references(
            request.references or [], request.num_frames
        )
        if request.height is None or request.width is None:
            if request.height is not None or request.width is not None:
                raise ValueError("height and width must be passed together")
            height, width = resolve_canvas_size(16, 9)
        else:
            height, width = request.height, request.width
        self._validate_canvas(height, width)
        geometry = self._resolve_geometry(height, width, num_frames)
        conditioning = self.conditioner.encode_ref2va(request.prompt, references)
        condition_video, condition_audio = self._encode_references(references, random)
        layout = build_ref2va_packed_sequence(
            conditioning.token_tags,
            references,
            *geometry,
            self.patch_size,
        )
        return (
            conditioning,
            layout,
            condition_video,
            condition_audio,
            num_frames,
            height,
            width,
            *geometry,
        )

    def _target_rows(
        self,
        request: MiniMaxH3GenerationRequest,
        random: _RandomStream,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
    ) -> tuple[mx.array, mx.array]:
        expected_video_shape = (
            1,
            self.video_vae.config.latent_channels,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        if request.latents is None:
            video = random.normal(expected_video_shape)
        else:
            video = request.latents.astype(mx.float32)
            if video.shape != expected_video_shape:
                raise ValueError(
                    f"latents must have shape {expected_video_shape}, got {video.shape}"
                )
        video_rows = patchify_video_latents(video, self.patch_size)
        expected_audio_shape = (
            2,
            self.audio_vae.config.latent_channels,
            num_audio_latents,
        )
        if request.audio_latents is None:
            audio_rows = random.normal(
                (
                    num_audio_latents * 2,
                    self.audio_vae.config.latent_channels,
                )
            )
        else:
            audio_rows = request.audio_latents.astype(mx.float32)
            if audio_rows.shape != expected_audio_shape:
                raise ValueError(
                    "audio_latents must have shape "
                    f"{expected_audio_shape}, got {audio_rows.shape}"
                )
            audio_rows = audio_rows.transpose(0, 2, 1).reshape(
                -1, self.audio_vae.config.latent_channels
            )
        return video_rows, audio_rows

    def _denoise(
        self,
        video_rows: mx.array,
        audio_rows: mx.array,
        conditioning: MiniMaxH3ConditioningOutput,
        layout,
        num_inference_steps: int,
    ) -> tuple[mx.array, mx.array]:
        self.scheduler.set_timesteps(num_inference_steps)
        self.audio_scheduler.set_timesteps(num_inference_steps)
        for video_timestep, audio_timestep in zip(
            self.scheduler.timesteps, self.audio_scheduler.timesteps
        ):
            timesteps, timestep_indices = build_row_timesteps(
                layout,
                float(video_timestep),
                float(audio_timestep),
                max(float(video_timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                1.0,
            )
            output = self.transformer(
                video_rows[None],
                audio_rows[None],
                conditioning.hidden_states,
                timesteps,
                timestep_indices,
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
            )
            video_start = layout.num_condition_video_rows
            audio_start = layout.num_condition_audio_rows
            updated_video = self.scheduler.step(
                output.sample[0, video_start:].astype(mx.float32),
                video_timestep,
                video_rows[video_start:],
            )
            updated_audio = self.audio_scheduler.step(
                output.audio_sample[0, audio_start:].astype(mx.float32),
                audio_timestep,
                audio_rows[audio_start:],
            )
            video_rows = mx.concatenate(
                [video_rows[:video_start], updated_video], axis=0
            )
            audio_rows = mx.concatenate(
                [audio_rows[:audio_start], updated_audio], axis=0
            )
            mx.eval(video_rows, audio_rows)
        return video_rows, audio_rows

    def _decode(
        self,
        video_rows: mx.array,
        audio_rows: mx.array,
        layout,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        output_type: str,
    ) -> tuple[mx.array, mx.array]:
        video_latents = unpatchify_video_tokens(
            video_rows[layout.num_condition_video_rows :],
            num_latent_frames,
            latent_height,
            latent_width,
            self.video_vae.config.latent_channels,
            self.patch_size,
        )
        video_mean = mx.array(self.video_vae.config.latents_mean, mx.float32).reshape(
            1, -1, 1, 1, 1
        )
        video_std = mx.array(self.video_vae.config.latents_std, mx.float32).reshape(
            1, -1, 1, 1, 1
        )
        video_latents = video_latents * video_std + video_mean

        audio_latents = unpack_audio_tokens(
            audio_rows[layout.num_condition_audio_rows :], num_audio_latents
        )
        audio_mean = mx.array(self.audio_vae.config.latents_mean, mx.float32).reshape(
            1, -1, 1
        )
        audio_std = mx.array(self.audio_vae.config.latents_std, mx.float32).reshape(
            1, -1, 1
        )
        audio_latents = audio_latents * audio_std + audio_mean
        if output_type == "latent":
            return video_latents, audio_latents

        video = self.video_vae.decode(video_latents).sample.astype(mx.float32)
        mean = mx.array(MINIMAX_H3_PIXEL_MEAN, mx.float32).reshape(1, -1, 1, 1, 1)
        std = mx.array(MINIMAX_H3_PIXEL_STD, mx.float32).reshape(1, -1, 1, 1, 1)
        video = mx.clip(video * std + mean, 0.0, 1.0)
        video = video.transpose(0, 2, 3, 4, 1)
        audio = self.audio_vae.decode(audio_latents).sample.transpose(1, 0, 2)
        return video, audio

    def generate(self, request: MiniMaxH3GenerationRequest) -> MiniMaxH3PipelineOutput:
        if not isinstance(request.prompt, str):
            raise ValueError("prompt must be one string")
        if request.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")
        if request.output_type not in ("array", "latent"):
            raise ValueError("output_type must be 'array' or 'latent'")
        has_references = request.references is not None
        if has_references != (self.partition == "ref2va"):
            raise ValueError(
                f"pipeline partition {self.partition} does not match the request"
            )
        random = _RandomStream(request.seed)
        if self.partition == "fl2va":
            (
                conditioning,
                layout,
                condition_video,
                num_frames,
                height,
                width,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
            ) = self._prepare_fl2va(request, random)
            condition_audio = None
        else:
            (
                conditioning,
                layout,
                condition_video,
                condition_audio,
                num_frames,
                height,
                width,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
            ) = self._prepare_ref2va(request, random)

        video_rows, audio_rows = self._target_rows(
            request,
            random,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
        )
        if condition_video is not None:
            video_rows = mx.concatenate([condition_video, video_rows])
        if condition_audio is not None:
            audio_rows = mx.concatenate([condition_audio, audio_rows])
        video_rows, audio_rows = self._denoise(
            video_rows,
            audio_rows,
            conditioning,
            layout,
            request.num_inference_steps,
        )
        video, audio = self._decode(
            video_rows,
            audio_rows,
            layout,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            request.output_type,
        )
        return MiniMaxH3PipelineOutput(
            video=video,
            audio=audio,
            sampling_rate=self.audio_vae.config.sampling_rate,
            fps=MINIMAX_H3_FPS,
            metadata={
                "partition": self.partition,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "seed": request.seed,
                "num_inference_steps": request.num_inference_steps,
                "video_latent_shape": (
                    1,
                    self.video_vae.config.latent_channels,
                    num_latent_frames,
                    latent_height,
                    latent_width,
                ),
                "audio_latent_shape": (
                    2,
                    self.audio_vae.config.latent_channels,
                    num_audio_latents,
                ),
            },
        )


__all__ = [
    "MiniMaxH3GenerationRequest",
    "MiniMaxH3Pipeline",
    "MiniMaxH3PipelineOutput",
]
