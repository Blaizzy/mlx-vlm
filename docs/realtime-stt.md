# Live-input speech transcription

`/v1/audio/transcriptions/realtime?model=<model>` accepts a single utterance over
WebSocket. It uses the existing audio model cache and worker, not a second
mlx-audio server. The inference router's bearer authentication also applies to
the WebSocket handshake.

This requires a model implementing the live-input `create_streaming_session`
contract (`feed`, `step`, `close`, `done`, `input_sample_rate`). Nemotron support
is proposed in https://github.com/Blaizzy/mlx-audio/pull/945; an unmodified
mlx-audio release without that contract is not supported. Other model factories
with the same name but a different contract are rejected.

After `session.created`, send JSON events:

```json
{"type":"input_audio_buffer.append","audio":"<base64 little-endian mono PCM16>"}
{"type":"input_audio_buffer.commit"}
```

Use the native sample rate announced in `session.created` (Nemotron: 16000 Hz).
Audio append events may be sent while inference is running. Output includes
`conversation.item.added`, `conversation.item.input_audio_transcription.delta`,
`input_audio_buffer.committed`, and
`conversation.item.input_audio_transcription.completed` (with `transcript`).
The socket closes after the final transcript. Open another socket for another
utterance. `session.cancel` or disconnect cancels the request.

`session.update` accepts only the same model, `input_audio_format: "pcm16"`,
the native `input_audio_sample_rate`, and `turn_detection: null`. There is no
automatic VAD, resampling, diarization, translation, or multi-turn session in
this endpoint. This is not a claim of full OpenAI Realtime protocol compatibility.

Limits: 128 KiB per JSON message, 64 queued input messages, 64 queued output
events, 30 seconds idle/input-stall timeout, 120 seconds per socket and per
utterance. Input waits for queue space to absorb encoder cold-start latency;
slow consumers eventually time out. Backend buffering limits still apply.
Send small chunks at playback pace; do not upload a whole recording at once.
WebSocket transports may buffer frames before application validation; configure
the ASGI server's WebSocket frame/queue limits for the deployment as well.

A live session exclusively reserves the audio worker. Another live or batch
audio request receives a busy error while reserved. A live session is rejected
when batch audio work is outstanding. Existing batch requests retain their
normal queue behavior. Model creation, feed, decode, close, and cancellation
all run on the audio worker thread.

## Validation

Run `pytest mlx_vlm/tests/test_server_audio.py
mlx_vlm/tests/test_server_stt_realtime.py mlx_vlm/tests/test_server_realtime.py`
on an MLX-capable host. The realtime STT tests use an in-memory fake model and
cover partial-before-commit, finalization, malformed input, authentication,
contention, disconnect, post-commit rejection, cold-step backpressure and
shutdown with pending requests. They do not download model weights.

A separate local socket proof through Nativ's unmodified Python overlay used
the candidate mlx-audio implementation and Nemotron 3.5 0.6B checkpoint
`e550040c0478027ed679b2b6b0d055502c103663`, with synthetic Portuguese and English
speech sent as 20 ms chunks. Both produced partials before commit. One run per
language measured first partial at 5.285 s / 2.214 s and commit-to-final at
0.260 s / 0.645 s respectively. These are functional observations, not latency
percentiles or a packaged Nativ app certification.
