import argparse
import asyncio
import json
import logging
import os
import threading
import time
import warnings
import wave

import aiohttp
import cv2
import numpy as np
import sounddevice as sd
import webrtcvad

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
    module="webrtcvad",
)

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageAudioStreamer:
    def __init__(
        self,
        api_url="http://localhost:8000/generate",
        model="mlx-community/gemma-3n-E2B-it-5bit",
        silence_threshold=0.03,
        silence_duration=2.0,
        sample_rate=16_000,
        frame_duration_ms=30,
        vad_mode=3,
        max_tokens=500,
        temperature=0.0,
        camera_index=0,
        capture_interval=2.0,
        enable_camera=True,
    ):
        self.api_url = api_url
        self.model = model
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.camera_index = camera_index
        self.capture_interval = capture_interval
        self.enable_camera = enable_camera

        self.vad = webrtcvad.Vad(vad_mode)
        self.input_audio_queue = asyncio.Queue(maxsize=50)
        self.session = None

        # Camera related attributes
        self.camera = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.camera_thread = None
        self.stop_camera = threading.Event()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        if self.enable_camera:
            self._start_camera()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.enable_camera:
            self._stop_camera()
        if self.session:
            await self.session.close()

    def _start_camera(self):
        """Initialize and start camera capture thread"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                self.enable_camera = False
                return

            # Start camera capture thread
            self.camera_thread = threading.Thread(target=self._camera_capture_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.enable_camera = False

    def _stop_camera(self):
        """Stop camera capture and release resources"""
        if self.camera_thread:
            self.stop_camera.set()
            self.camera_thread.join(timeout=2.0)

        if self.camera:
            self.camera.release()
            self.camera = None
        logger.info("Camera released")

    def _camera_capture_loop(self):
        """Continuously capture frames from camera"""
        last_capture_time = 0

        while not self.stop_camera.is_set():
            current_time = time.time()

            # Capture frame at specified interval
            if current_time - last_capture_time >= self.capture_interval:
                ret, frame = self.camera.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    last_capture_time = current_time
                else:
                    logger.error("Failed to capture frame from camera")

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.1)

    def _save_latest_frame(self, filename):
        """Save the latest captured frame to file"""
        with self.frame_lock:
            if self.latest_frame is not None:
                cv2.imwrite(filename, self.latest_frame)
                return True
        return False

    def _is_silent(self, audio_data):
        """Energy-based silence detection"""
        if isinstance(audio_data, bytes):
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0
        else:
            audio_np = audio_data.astype(np.float32)

        energy = np.linalg.norm(audio_np) / np.sqrt(audio_np.size)
        return energy < self.silence_threshold

    def _voice_activity_detection(self, frame):
        """Voice activity detection using WebRTC VAD with fallback"""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except ValueError:
            return not self._is_silent(frame)

    def _save_audio_to_wav(self, audio_data, filename):
        """Save audio data to WAV file with validation"""
        if len(audio_data) == 0:
            logger.error("Empty audio data!")
            return

        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)

    async def _send_transcription_request(
        self,
        audio_file_path: str,
        image_file_path: str = None,
        prompt: str = "Transcribe this text in English:",
    ):
        """Send audio file to transcription API and handle streaming response"""

        # Use absolute path for the API call
        absolute_audio_path = os.path.abspath(audio_file_path)

        # Prepare image paths if available
        image_paths = []
        if image_file_path and os.path.exists(image_file_path):
            absolute_image_path = os.path.abspath(image_file_path)
            image_paths.append(absolute_image_path)

        payload = {
            "model": self.model,
            "image": image_paths,
            "audio": [absolute_audio_path],
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            async with self.session.post(self.api_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"API request failed with status {response.status}: {error_text}"
                    )
                    return

                transcription = ""

                async for line in response.content:
                    if line:
                        try:
                            # Parse streaming JSON response
                            line_text = line.decode("utf-8").strip()

                            if line_text.startswith("data: "):
                                json_str = line_text[6:]  # Remove 'data: ' prefix

                                if json_str == "[DONE]":
                                    break

                                data = json.loads(json_str)

                                # The API returns chunks directly in the 'chunk' field
                                if "chunk" in data:
                                    chunk_content = data["chunk"]
                                    if chunk_content:  # Only add non-empty chunks
                                        transcription += chunk_content
                                        print(chunk_content, end="", flush=True)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Error processing stream: {e}")
                            continue

                if transcription:
                    print(" ", end="", flush=True)
                    return transcription.strip()

        except Exception as e:
            logger.error(f"Transcription request failed: {e}")

    def _sd_callback(self, indata, frames, _time, status):
        """Sounddevice callback for audio input"""
        data = indata.reshape(-1).tobytes()

        def _enqueue():
            try:
                self.input_audio_queue.put_nowait(data)
            except asyncio.QueueFull:
                return

        # Use the stored event loop reference instead of get_event_loop()
        self.loop.call_soon_threadsafe(_enqueue)

    async def stream_microphone_transcription(
        self, prompt: str = "Transcribe this text in English:"
    ):
        """Stream transcription from microphone input"""
        # Store the current event loop for use in the callback
        self.loop = asyncio.get_running_loop()

        frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        frames_until_silence = int(
            self.silence_duration * 1000 / self.frame_duration_ms
        )

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=frame_size,
            channels=1,
            dtype="int16",
            callback=self._sd_callback,
        )

        stream.start()
        logger.info("Listening for voice input... (Press Ctrl+C to stop)")

        frames = []
        silent_frames = 0
        speaking_detected = False
        audio_counter = 0

        try:
            while True:
                frame = await self.input_audio_queue.get()
                is_speech = self._voice_activity_detection(frame)

                if is_speech:
                    if not speaking_detected:
                        logger.info("Speech detected, recording...")
                    speaking_detected = True
                    silent_frames = 0
                    frames.append(frame)
                elif speaking_detected:
                    silent_frames += 1
                    frames.append(frame)

                    if silent_frames > frames_until_silence:
                        if frames:
                            # Save audio to current directory instead of temp
                            audio_data = b"".join(frames)

                            # Save in current working directory with simple name
                            temp_audio_filename = f"debug_audio_{audio_counter}.wav"
                            full_audio_path = os.path.abspath(temp_audio_filename)

                            self._save_audio_to_wav(audio_data, full_audio_path)

                            # Save camera frame if available
                            temp_image_filename = None
                            if self.enable_camera:
                                temp_image_filename = f"debug_frame_{audio_counter}.jpg"
                                if self._save_latest_frame(temp_image_filename):
                                    logger.debug(
                                        f"Saved camera frame: {temp_image_filename}"
                                    )
                                else:
                                    temp_image_filename = None

                            try:
                                # Send for transcription with both audio and image
                                await self._send_transcription_request(
                                    temp_audio_filename, temp_image_filename, prompt
                                )
                            finally:
                                # Clean up temporary files after transcription
                                try:
                                    if temp_audio_filename:
                                        os.remove(temp_audio_filename)
                                    if temp_image_filename:
                                        os.remove(temp_image_filename)
                                except Exception as e:
                                    logger.error(
                                        f"Failed to remove temporary files: {e}"
                                    )

                            audio_counter += 1

                        frames = []
                        speaking_detected = False
                        silent_frames = 0
                        logger.info("Ready for next input...")

        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Stopping microphone transcription...")
        finally:
            stream.stop()
            stream.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Audio Transcription Microphone Streamer"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:8000/generate",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3n-E2B-it-5bit",
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the following speech segment in English:",
        help="Transcription prompt",
    )
    parser.add_argument(
        "--silence_duration",
        type=float,
        default=2.0,
        help="Duration of silence before processing audio (seconds)",
    )
    parser.add_argument(
        "--silence_threshold",
        type=float,
        default=0.03,
        help="Energy threshold for silence detection",
    )
    parser.add_argument(
        "--vad_mode",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="WebRTC VAD aggressiveness (0-3, higher = more aggressive)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="Maximum tokens for transcription"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--camera", action="store_true", help="Enable camera capture alongside audio"
    )
    parser.add_argument(
        "--camera_index", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--capture_interval",
        type=float,
        default=2.0,
        help="Interval between camera captures in seconds (default: 2.0)",
    )

    args = parser.parse_args()

    print("\033[92mImage & Audio Streamer\033[0m")
    print("\033[92mModel: \033[0m", args.model)
    print("\033[92mAPI URL: \033[0m", args.api_url)
    print("\033[92mSilence Threshold: \033[0m", args.silence_threshold)
    print("\033[92mSilence Duration: \033[0m", args.silence_duration)
    print("\033[92mVAD Mode: \033[0m", args.vad_mode)
    print("\033[92mMax Tokens: \033[0m", args.max_tokens)
    print("\033[92mTemperature: \033[0m", args.temperature)
    print("\033[92mCamera Enabled: \033[0m", args.camera)
    if args.camera:
        print("\033[92mCamera Index: \033[0m", args.camera_index)
        print("\033[92mCapture Interval: \033[0m", args.capture_interval)
    print("\033[92mPrompt: \033[0m", args.prompt)

    async with ImageAudioStreamer(
        api_url=args.api_url,
        model=args.model,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        vad_mode=args.vad_mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        camera_index=args.camera_index,
        capture_interval=args.capture_interval,
        enable_camera=args.camera,
    ) as streamer:
        await streamer.stream_microphone_transcription(args.prompt)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
