"""Speech output for omni models, using the shared text generation path."""

from dataclasses import dataclass, fields, replace
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .common import GenerationResult, generation_stream, wired_limit
from .types import ProcessorLike


@dataclass
class AudioGenerationResult(GenerationResult):
    """Generated speech as a mono waveform and its sample rate in Hz."""

    audio: Optional[mx.array] = None
    sample_rate: int = 24000
    audio_tokens: Optional[mx.array] = None
    path: Optional[Path] = None

    def to_wav_bytes(self) -> bytes:
        from mlx_audio.audio_io import write

        if self.audio is None or self.audio.size == 0:
            raise ValueError("No audio was generated.")
        buffer = BytesIO()
        write(
            buffer,
            np.asarray(self.audio, dtype=np.float32),
            samplerate=self.sample_rate,
            format="wav",
        )
        return buffer.getvalue()


def save_audio(result: AudioGenerationResult, path: Union[str, Path]) -> Path:
    """Save generated speech as a WAV file and record its path on the result."""
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError("Audio output must use a .wav extension.")
    data = result.to_wav_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    result.path = path
    return path


def is_audio_generation_model(model: nn.Module) -> bool:
    """Check local model capabilities without downloading any metadata."""
    return callable(getattr(model, "generate_audio", None)) and getattr(
        model, "supports_audio_generation", True
    )


def generate_audio(
    model: nn.Module,
    processor: ProcessorLike,
    prompt: str,
    image: Union[str, List[str], None] = None,
    audio: Union[str, List[str], None] = None,
    video: Union[str, List[str], None] = None,
    verbose: bool = False,
    output_audio_path: Union[str, Path, None] = None,
    ref_audio_path: Optional[str] = None,
    **kwargs,
) -> AudioGenerationResult:
    """Generate a spoken response to a formatted chat prompt.

    Load the omni model with :func:`mlx_vlm.load` and format its prompt with
    ``apply_chat_template(..., use_tts_template=True)``. Text sampling options
    are shared with ``generate``; options prefixed with ``tts_`` go only to
    the speech model. ``audio`` is an input to understand; ``ref_audio_path``
    supplies the reference voice to both the model's prompt and its codec.
    The waveform is returned even without a file path. MiniCPM-o currently
    requires a reference WAV (or audio input).
    """
    from .dispatch import _prepare_generation_inputs, generate

    if not is_audio_generation_model(model):
        raise ValueError(
            f"{type(model).__name__} does not support audio generation with these "
            "weights. Use a checkpoint containing the TTS module."
        )
    if (
        output_audio_path is not None
        and Path(output_audio_path).suffix.lower() != ".wav"
    ):
        raise ValueError("Audio output must use a .wav extension.")
    validate = getattr(model, "validate_audio_generation", None)
    if callable(validate):
        validate(audio=audio, ref_audio_path=ref_audio_path)

    prepare_reference = getattr(processor, "prepare_audio_generation", None)
    if callable(prepare_reference):
        if kwargs.get("input_ids") is not None:
            raise ValueError(
                "Automatic reference-voice conditioning requires a formatted "
                "prompt, not precomputed input_ids."
            )
        prompt, audio = prepare_reference(
            prompt, audio=audio, ref_audio_path=ref_audio_path
        )

    speech_kwargs = {
        key: kwargs.pop(key) for key in list(kwargs) if key.startswith("tts_")
    }
    input_ids, pixel_values, mask, data_kwargs = _prepare_generation_inputs(
        model, processor, prompt, image, audio, video, kwargs
    )
    text_result = generate(
        model,
        processor,
        prompt,
        image,
        audio,
        video,
        verbose=verbose,
        input_ids=input_ids,
        pixel_values=pixel_values,
        mask=mask,
        **kwargs,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        result = model.generate_audio(
            input_ids=input_ids,
            generated_tokens=text_result.token_ids or [],
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            mask=mask,
            audio=audio,
            ref_audio_path=ref_audio_path,
            **data_kwargs,
            **speech_kwargs,
        )
        mx.eval(result.audio, result.audio_tokens)
    spoken_text = result.text or text_result.text
    result = replace(
        result,
        **{
            field.name: getattr(text_result, field.name)
            for field in fields(GenerationResult)
        },
    )
    result.text = spoken_text
    result.peak_memory = mx.get_peak_memory() / 1e9
    if output_audio_path is not None:
        save_audio(result, output_audio_path)
    return result
