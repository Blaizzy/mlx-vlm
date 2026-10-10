# NVIDIA NemotronLabs VoiceChat

NemotronLabs VoiceChat is a full-duplex speech model that listens, transcribes,
responds with text, and synthesizes aligned speech on a continuous audio
timeline. This MLX implementation supports online inference through the
`/v1/realtime` WebSocket endpoint and a stateful Python API, plus offline WAV
inference.

Input audio is 16 kHz mono. Generated audio is 22,050 Hz mono using the model's
built-in `Aria` voice.

## Supported Models

| Model | Precision |
|---|---|---|
| `mlx-community/NemotronLabs-VoiceChat-11B-bf16` | BF16 |
| `mlx-community/NemotronLabs-VoiceChat-11B-8bit` | 8-bit |
| `mlx-community/NemotronLabs-VoiceChat-11B-4bit` | 4-bit |

## Install

```sh
pip install -U mlx-vlm
```

## Realtime WebSocket API

Start the normal `mlx-vlm` server:

```sh
mlx_vlm.server --host 127.0.0.1 --port 8080
```

VoiceChat clients connect to:

```text
ws://127.0.0.1:8080/v1/realtime
```

The server sends `session.created` after accepting the connection. Configure
the session before sending audio:

```json
{
  "type": "session.update",
  "session": {
    "model": "mlx-community/NemotronLabs-VoiceChat-11B-8bit",
    "system_prompt": "Be concise and answer in one sentence.",
    "seed": 0
  }
}
```

After `session.updated`, send any number of base64-encoded PCM16 chunks. Chunk
boundaries may be arbitrary; the server advances the model on complete 1,280
sample (80 ms) frames.

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 PCM16>",
  "sample_rate": 16000
}
```

Commit the input when the session is finished:

```json
{
  "type": "input_audio_buffer.commit",
  "pad_partial": true
}
```

The server emits aligned events as inference advances:

| Event | Contents |
|---|---|
| `response.text.delta` | Assistant token, incremental text, and cumulative text |
| `conversation.item.input_audio_transcription.delta` | Incremental and cumulative user transcript |
| `response.function.delta` | Raw function-channel token and text delta |
| `response.audio.delta` | Base64 PCM16 audio at 22,050 Hz plus raw audio codes |
| `response.done` | The committed session has finished |
| `response.cancelled` | The session was cancelled |
| `error` | Request or inference error details |

Send `session.cancel` or `response.cancel` to stop without committing. A client
may also send `session.ping` and receive `session.pong`.

### Microphone and speakers

The live microphone example also requires PortAudio through `sounddevice`:

```sh
pip install -U sounddevice
```

For a live microphone and speaker loop:

```sh
python examples/nemotron_voicechat_microphone.py \
  mlx-community/NemotronLabs-VoiceChat-11B-8bit
```

Use `--list-devices`, `--input-device`, and `--output-device` to select audio
devices. Headphones are strongly recommended: the example keeps listening while
the assistant speaks and does not perform acoustic echo cancellation.

## Python API

Load the model and processor through the standard `mlx_vlm.load` interface,
then create a model-specific VoiceChat session.

### Online inference

The stateful API accepts floating-point mono PCM with arbitrary chunk
boundaries. Text fields on events are cumulative; `delta` contains the
incremental update.

```python
import numpy as np
from mlx_audio.audio_io import write as write_audio
from mlx_audio.stt.utils import load_audio

from mlx_vlm import load

model, processor = load(
    "mlx-community/NemotronLabs-VoiceChat-11B-8bit"
)
voicechat = model.create_session(processor)
stream = voicechat.create_streaming_session(
    system_prompt="Be concise and answer in one sentence.",
    seed=0,
)

input_audio = load_audio("input.wav", sr=16_000).squeeze()
audio_chunks = []

def handle(events):
    for event in events:
        if event.kind == "assistant_text_delta":
            print(event.delta or "", end="", flush=True)
        elif event.kind == "user_transcript_delta":
            print(f"\n[user] {event.text}")
        elif event.kind == "function_delta":
            print(f"\n[function] {event.delta}")
        elif event.kind == "audio":
            audio_chunks.append(np.asarray(event.samples, dtype=np.float32))

for offset in range(0, input_audio.shape[0], 1_280):
    handle(
        stream.push_audio(
            input_audio[offset : offset + 1_280],
            sample_rate=16_000,
        )
    )
handle(stream.flush(pad_partial=True))

response_audio = (
    np.concatenate(audio_chunks)
    if audio_chunks
    else np.zeros(0, dtype=np.float32)
)
write_audio("response.wav", response_audio, 22_050)
```

Online sessions keep persistent log-mel, FastConformer, Nemotron-H, EAR-TTS,
and codec state by default. This bounded per-frame path is the recommended mode
for interactive applications. Whether it keeps pace with wall clock depends on
the model precision and host hardware.

For diagnostic comparisons, language and perception caches can be disabled
independently:

```python
stream = voicechat.create_streaming_session(
    use_language_cache=False,
    use_perception_cache=False,
)
```

### Offline inference

```python
result = voicechat.generate(
    "input.wav",
    system_prompt="Be concise and answer in one sentence.",
    extra_decoding_seconds=3,
    seed=0,
)

print(result.user_transcript)
print(result.text)
# result.audio is mono float PCM at result.sample_rate (22,050 Hz).
```

The offline path uses full-history Nemotron-H inference by default.

## Notes

- The model is continuously duplex: silence and overlapping speech remain part
  of its timeline. The server does not use VAD to gate input.
- `/v1/realtime` currently allows one active VoiceChat WebSocket at a time.
- Arbitrary voice cloning is not supported; inference uses the checkpoint's
  built-in `Aria` voice.
- Remote Hugging Face model code is not executed.
