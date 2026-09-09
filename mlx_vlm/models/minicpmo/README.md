# MiniCPM-o (MiniCPM-o-4_5)

MiniCPM-o is an omni model that supports text, image, and audio understanding.

This MLX-VLM integration includes:
- Custom MiniCPM-o processor registration
- Image + audio input preprocessing
- Prompt placeholder handling for `<image>` and `<audio>`

## Model

- Hugging Face ID: `openbmb/MiniCPM-o-4_5`
- Remote tokenizer/processor code is ported in-tree, so `--trust-remote-code` is not required.

## Install

```sh
pip install -U mlx-vlm
```

## CLI

### Image understanding

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --image /path/to/image.jpg \
  --prompt "Describe this image briefly." \
  --max-tokens 128 \
  --temperature 0
```

### Audio understanding

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --audio /path/to/audio.wav \
  --prompt "Describe this audio briefly." \
  --max-tokens 256 \
  --temperature 0
```

### Multi-modal (image + audio)

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --image /path/to/image.jpg \
  --audio /path/to/audio.wav \
  --prompt "Describe what you see and hear." \
  --max-tokens 256 \
  --temperature 0
```

### Enable thinking in chat template

Thinking is disabled by default. To enable it:

```sh
uv run mlx_vlm.generate \
  --model openbmb/MiniCPM-o-4_5 \
  --audio /path/to/audio.wav \
  --prompt "Describe this audio briefly." \
  --enable-thinking \
  --max-tokens 256 \
  --temperature 0
```

## Python

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load(
    "openbmb/MiniCPM-o-4_5",
)

image = ["/path/to/image.jpg"]
audio = ["/path/to/audio.wav"]
prompt = "Summarize the visual and audio content."

formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=len(image),
    num_audios=len(audio),
    enable_thinking=False,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=image,
    audio=audio,
    max_tokens=256,
    temperature=0.0,
)
print(result.text)
```

## Notes

- You usually should not manually add `<image>` or `<audio>` markers when using `apply_chat_template`.
- Multiple `--audio` inputs are processed in the order supplied.

## Speech output

MiniCPM-o 4.5 can generate a spoken response using `--output-modality audio`.
The CLI formats the TTS prompt, generates the response text, and synthesizes a
24 kHz mono WAV file.

```sh
python -m mlx_vlm generate \
  --model openbmb/MiniCPM-o-4_5 \
  --output-modality audio \
  --prompt "Say hello in one short sentence." \
  --ref-audio /path/to/reference.wav \
  --output speech.wav \
  --max-tokens 256 \
  --temperature 0 \
  --seed 7
```

`python -m mlx_vlm generate_audio` is an alias for the same path. Supply a short,
clean voice recording with `--ref-audio`. If it is omitted, the first `--audio`
input supplies the reference voice. The reference conditions both the speech
codec and a voice-cloning system prompt, following the upstream MiniCPM-o
recipe. This lets the language-model hidden states used by TTS carry the
reference voice and style. Existing system instructions and separate image or
audio inputs are preserved. Pass a formatted prompt to `generate_audio()` so
it can insert the reference before tokenization; precomputed `input_ids` are
not supported by this automatic conditioning path.

The first synthesis loads the separate `mlx-community/Step-Audio-2-token2wav`
codec and its S3 tokenizer. Decoding runs in MLX, without PyTorch or remote
Python code. This path currently supports a single response at a time, with
non-streaming waveform output; `--chat` is not supported.

Text generation uses the usual sampling arguments. Speech sampling can be tuned
independently with `--gen-kwargs`, for example:

```sh
--gen-kwargs '{"tts_max_tokens": 1024, "tts_temperature": 0.8}'
```

The supported speech options are `tts_max_tokens` (2048), `tts_temperature` (0.8),
`tts_top_p` (0.85), `tts_top_k` (25), and `tts_repetition_penalty` (1.05).
Parenthesized values are the defaults for MiniCPM-o 4.5. The text budget must be
large enough to finish the spoken response and generate its `<|tts_eos|>` marker.

### Python speech API

```python
from mlx_vlm import apply_chat_template, generate_audio, load, save_audio

model, processor = load("openbmb/MiniCPM-o-4_5")
prompt = apply_chat_template(
    processor,
    model.config,
    "Say hello in one short sentence.",
    enable_thinking=False,
    use_tts_template=True,
)
result = generate_audio(
    model,
    processor,
    prompt,
    ref_audio_path="/path/to/reference.wav",
    max_tokens=256,
    temperature=0,
)
print(result.text)          # Spoken response, without TTS control tokens
print(result.audio.shape)  # MLX mono waveform, available without writing a file
print(result.sample_rate)  # 24000
save_audio(result, "speech.wav")
# result.to_wav_bytes() returns an in-memory WAV.
```

`AudioGenerationResult` includes the text generation statistics and token IDs,
the speech codec tokens, the waveform, sample rate, and saved `path` (if any).
Text throughput statistics describe the text stage; `peak_memory` includes
speech synthesis. `generate_audio(..., output_audio_path="speech.wav")` saves
the waveform directly.
