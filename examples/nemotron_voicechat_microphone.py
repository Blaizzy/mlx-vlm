"""Talk to Nemotron VoiceChat through the local mlx-vlm WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import signal
import sys
import threading
from contextlib import suppress
from functools import partial
from typing import Any

import websockets


class PlaybackBuffer:
    """Thread-safe PCM16 buffer shared with PortAudio's output callback."""

    def __init__(self):
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def append(self, pcm16: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm16)

    def take(self, size: int) -> bytes:
        with self._lock:
            count = min(size, len(self._buffer))
            result = bytes(self._buffer[:count])
            del self._buffer[:count]
        if count < size:
            result += bytes(size - count)
        return result

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._buffer)


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_latency(value: str) -> str | float:
    if value in {"low", "high"}:
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "latency must be low, high, or seconds"
        ) from exc


async def run(args, sounddevice) -> None:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    input_queue: asyncio.Queue[bytes] = asyncio.Queue()
    playback = PlaybackBuffer()
    capture_active = threading.Event()
    capture_active.set()
    loop = asyncio.get_running_loop()
    warned_backlog = False

    def report_status(direction: str, status: Any) -> None:
        if status:
            loop.call_soon_threadsafe(
                partial(
                    print,
                    f"\n[{direction}] {status}",
                    file=sys.stderr,
                    flush=True,
                )
            )

    def input_callback(indata, frames, time_info, status) -> None:
        del frames, time_info
        report_status("microphone", status)
        if capture_active.is_set():
            loop.call_soon_threadsafe(input_queue.put_nowait, bytes(indata))

    def output_callback(outdata, frames, time_info, status) -> None:
        del frames, time_info
        report_status("speaker", status)
        outdata[:] = playback.take(len(outdata))

    def request_stop() -> None:
        if capture_active.is_set():
            capture_active.clear()
            print("\nStopping microphone and draining responses...", flush=True)

    registered_signals = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop)
            registered_signals.append(signum)
        except NotImplementedError:
            pass

    timer_task = None

    async def stop_after_duration() -> None:
        await asyncio.sleep(args.duration)
        request_stop()

    input_device = parse_device(args.input_device)
    output_device = parse_device(args.output_device)

    try:
        async with websockets.connect(
            args.url,
            additional_headers=headers,
            max_queue=None,
        ) as socket:
            created = json.loads(await socket.recv())
            if created.get("type") != "session.created":
                raise RuntimeError(created)
            session_config = {
                "model": args.model,
                "system_prompt": args.system_prompt,
                "seed": args.seed,
            }
            if args.max_streaming_seconds is not None:
                session_config["max_streaming_seconds"] = args.max_streaming_seconds
            await socket.send(
                json.dumps({"type": "session.update", "session": session_config})
            )
            updated = json.loads(await socket.recv())
            if updated.get("type") != "session.updated":
                raise RuntimeError(updated)

            session = updated["session"]
            input_rate = int(session["input_audio_format"]["sample_rate"])
            output_rate = int(session["output_audio_format"]["sample_rate"])
            input_blocksize = int(session["frame_samples"])
            output_blocksize = round(output_rate * input_blocksize / input_rate)

            sounddevice.check_input_settings(
                device=input_device,
                channels=1,
                dtype="int16",
                samplerate=input_rate,
            )
            sounddevice.check_output_settings(
                device=output_device,
                channels=1,
                dtype="int16",
                samplerate=output_rate,
            )

            async def send_microphone() -> None:
                nonlocal warned_backlog
                while True:
                    try:
                        chunk = await asyncio.wait_for(input_queue.get(), timeout=0.1)
                    except TimeoutError:
                        if not capture_active.is_set():
                            break
                        continue

                    await socket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("ascii"),
                                "sample_rate": input_rate,
                            }
                        )
                    )
                    input_queue.task_done()
                    if input_queue.qsize() >= args.backlog_warning_frames:
                        if not warned_backlog:
                            delay = input_queue.qsize() * input_blocksize / input_rate
                            print(
                                f"\n[warning] microphone backlog is {delay:.1f}s; "
                                "this machine is not keeping up with wall clock",
                                file=sys.stderr,
                                flush=True,
                            )
                            warned_backlog = True
                    else:
                        warned_backlog = False

                await socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.commit",
                            "pad_partial": True,
                        }
                    )
                )

            async def receive_events() -> None:
                while True:
                    event = json.loads(await socket.recv())
                    event_type = event.get("type")
                    if event_type == "response.text.delta":
                        print(event.get("delta", ""), end="", flush=True)
                    elif (
                        event_type
                        == "conversation.item.input_audio_transcription.delta"
                    ):
                        print(f"\n[user] {event.get('transcript', '')}", flush=True)
                    elif event_type == "response.function.delta":
                        print(f"\n[function] {event.get('text', '')}", flush=True)
                    elif event_type == "response.audio.delta":
                        if int(event["sample_rate"]) != output_rate:
                            raise RuntimeError(
                                "server changed output sample rate during the session"
                            )
                        playback.append(base64.b64decode(event["delta"]))
                    elif event_type == "error":
                        raise RuntimeError(event["error"]["message"])
                    elif event_type == "response.done":
                        return

            print(
                f"Listening at {input_rate} Hz and playing at {output_rate} Hz. "
                "Press Ctrl-C to stop. Headphones are recommended.",
                flush=True,
            )
            if args.duration is not None:
                timer_task = asyncio.create_task(stop_after_duration())
            with (
                sounddevice.RawInputStream(
                    samplerate=input_rate,
                    blocksize=input_blocksize,
                    device=input_device,
                    channels=1,
                    dtype="int16",
                    latency=args.latency,
                    callback=input_callback,
                ),
                sounddevice.RawOutputStream(
                    samplerate=output_rate,
                    blocksize=output_blocksize,
                    device=output_device,
                    channels=1,
                    dtype="int16",
                    latency=args.latency,
                    callback=output_callback,
                ),
            ):
                await asyncio.gather(send_microphone(), receive_events())

                deadline = loop.time() + args.drain_timeout
                while playback.buffered_bytes and loop.time() < deadline:
                    await asyncio.sleep(0.05)
                if playback.buffered_bytes:
                    print(
                        "\n[warning] timed out while draining speaker audio",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    await asyncio.sleep(output_blocksize / output_rate)
    finally:
        capture_active.clear()
        if timer_task is not None:
            timer_task.cancel()
            with suppress(asyncio.CancelledError):
                await timer_task
        for signum in registered_signals:
            loop.remove_signal_handler(signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        metavar="MODEL_REPO_ID",
        help=(
            "Hugging Face model repository ID, for example "
            "mlx-community/NemotronLabs-VoiceChat-11B-8bit"
        ),
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8080/v1/realtime")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--latency", type=parse_latency, default="low")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-streaming-seconds", type=float)
    parser.add_argument("--backlog-warning-frames", type=int, default=25)
    parser.add_argument("--drain-timeout", type=float, default=10.0)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    try:
        import sounddevice
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is required; install it with `pip install 'mlx-vlm[realtime]'`"
        ) from exc

    if args.list_devices:
        print(sounddevice.query_devices())
        return
    if not args.model:
        parser.error("MODEL_REPO_ID is required unless --list-devices is used")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.max_streaming_seconds is not None and args.max_streaming_seconds <= 0:
        parser.error("--max-streaming-seconds must be positive")
    if args.backlog_warning_frames <= 0:
        parser.error("--backlog-warning-frames must be positive")
    if args.drain_timeout < 0:
        parser.error("--drain-timeout must be non-negative")

    asyncio.run(run(args, sounddevice))


if __name__ == "__main__":
    main()
