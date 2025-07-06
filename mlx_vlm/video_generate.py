from __future__ import annotations

import argparse
import base64
import logging
import math
import os
import sys
import time
from io import BytesIO
from typing import List

import cv2
import mlx.core as mx
import numpy as np
import requests
from PIL import Image

from .generate import generate
from .utils import load, load_image, process_inputs_with_fallback

# This is a beta version of the video generation script.
# It is not fully tested and may not work as expected.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

logger.info(
    "This is a beta version of the video understanding. It may not work as expected."
)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(
            pil_image, mask=pil_image.split()[3]
        )  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(
    ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR
) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """Calculate the number of frames for the video to be used as model inputs.

    Either a fixed 'nframes' is provided in ele or 'fps' is used to calculate how many frames to sample.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should be in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def load_video(
    ele: dict,
) -> (np.ndarray, float):
    """
    Read video using cv2.VideoCapture.

    The video is read as a NumPy array with shape (T, C, H, W) where T is the number of frames,
    C is the number of channels, and H, W are the frame dimensions.
    """
    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 1.0  # default to 1.0 if fps returns 0
    st = time.time()
    logger.info(
        f"numpy reader: video_path={video_path}, total_frames={total_frames}, video_fps={video_fps}, time={time.time()-st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames read from the video.")
    # Stack frames into a numpy array: (T, H, W, C)
    video_np = np.stack(frames, axis=0)
    # Rearrange to (T, C, H, W)
    video_np = np.transpose(video_np, (0, 3, 1, 2))
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video_np, sample_fps


def fetch_video(
    ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False
) -> np.ndarray | list[Image.Image]:
    if isinstance(ele["video"], str):
        video, sample_fps = load_video(ele)
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        if max_pixels_supposed > max_pixels:
            logger.warning(
                f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
            )
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        # Resize each frame using OpenCV (similar to torchvision.transforms.functional.resize with BICUBIC)
        resized_frames = []
        # video is (T, C, H, W) so we need to process each frame
        for frame in video:
            # Rearrange from (C, H, W) to (H, W, C)
            frame_np = np.transpose(frame, (1, 2, 0))
            # cv2.resize expects size as (width, height)
            resized = cv2.resize(
                frame_np, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC
            )
            # Convert back to (C, H, W)
            resized = np.transpose(resized, (2, 0, 1))
            resized_frames.append(resized)
        video = np.stack(resized_frames, axis=0).astype(np.float32)
        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        # Assume video is provided as a list/tuple of image objects.
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image(
                {"image": video_element, **process_info}, size_factor=image_factor
            )
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[
    list[Image.Image] | None, list[np.ndarray | list[Image.Image]] | None, dict | None
]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info, return_video_sample_fps=True
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("Content must include image, image_url, or video.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {"fps": video_sample_fps_list}
    return image_inputs, video_inputs


class VideoFrameExtractor:
    def __init__(self, max_frames: int = 50):
        self.max_frames = max_frames

    def resize_and_center_crop(
        self, image: Image.Image, target_size: int
    ) -> Image.Image:
        # Get current dimensions
        width, height = image.size

        # Calculate new dimensions keeping aspect ratio
        if width < height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size

        return image.crop((left, top, right, bottom))

    def extract_frames(self, video_path: str) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Calculate frame indices to extract (1fps)
        frame_indices = list(range(0, total_frames, fps))

        # If we have more frames than max_frames, sample evenly
        if len(frame_indices) > self.max_frames:
            indices = np.linspace(0, len(frame_indices) - 1, self.max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                pil_image = self.resize_and_center_crop(pil_image, 384)
                frames.append(pil_image)

        cap.release()
        return frames


def is_video_model(model):
    return hasattr(model.config, "video_token_id") or hasattr(
        model.config, "video_token_index"
    )


def is_video_file(video_path: List[str]) -> bool:
    video_extensions = [".mp4", ".avi", ".mov"]
    for path in video_path:
        if not any(path.endswith(ext) for ext in video_extensions):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Video Description CLI")
    parser.add_argument(
        "--video", type=str, nargs="+", required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        nargs=2,
        default=224 * 224,
        help="Maximum number of pixels",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames"
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second")
    parser.add_argument(
        "--prompt", default="Describe this video.", help="Text prompt for the model"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        help="Select the model to use",
    )
    parser.add_argument("--verbose", action="store_false", help="Print verbose output")

    args = parser.parse_args()

    print(f"\033[32mLoading model:\033[0m {args.model}")
    model, processor = load(args.model)

    # Validate the model
    if not is_video_model(model):
        logger.warning(
            "Warning: The model selected doesn't natively support video inputs. Performance may be degraded."
        )

    if isinstance(args.max_pixels, tuple) or isinstance(args.max_pixels, list):
        max_pixels = args.max_pixels[0] * args.max_pixels[1]
    else:
        max_pixels = args.max_pixels

    kwargs = {}
    if is_video_model(model):

        # Check if video is image or video
        if is_video_file(args.video):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": args.video[0],
                            "max_pixels": max_pixels,
                            "fps": args.fps,
                        },
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in args.video],
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, fps = process_vision_info(messages, True)

        if args.max_frames is not None:
            video_inputs = video_inputs[: args.max_frames]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = mx.array(inputs["input_ids"])
        pixel_values = inputs.get(
            "pixel_values_videos", inputs.get("pixel_values", None)
        )
        if pixel_values is None:
            raise ValueError("Please provide a valid video or image input.")
        pixel_values = mx.array(pixel_values)

        mask = mx.array(inputs["attention_mask"])
        if inputs.get("video_grid_thw", None) is not None:
            kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])
        if inputs.get("image_grid_thw", None) is not None:
            kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])

    else:
        if is_video_file(args.video):
            if len(args.video) > 1:
                raise ValueError("Only one video is supported for video models.")
            else:
                frame_extractor = VideoFrameExtractor(args.max_frames)
                frames = frame_extractor.extract_frames(args.video[0])
        else:
            frames = [load_image(image) for image in args.video]

        # Create prompt with frames
        image_tokens = [{"type": "image"} for _ in range(len(frames))]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    *image_tokens,
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Configure processor for video frames
        processor.image_processor.size = (
            args.max_pixels
            if isinstance(args.max_pixels, tuple)
            else (args.max_pixels, args.max_pixels)
        )
        if hasattr(processor.image_processor, "do_resize"):
            processor.image_processor.do_resize = False
        if hasattr(processor.image_processor, "do_image_splitting"):
            processor.image_processor.do_image_splitting = False

        # Process inputs
        inputs = process_inputs_with_fallback(
            processor,
            images=[img for img in frames],
            prompts=text,
        )

        input_ids = mx.array(inputs["input_ids"])
        pixel_values = mx.array(inputs["pixel_values"])
        mask = mx.array(inputs["attention_mask"])
        for key, value in inputs.items():
            if key not in [
                "input_ids",
                "pixel_values",
                "attention_mask",
            ] and not isinstance(value, (str, list)):
                kwargs[key] = mx.array(value)

    logger.info("\033[32mGenerating response...\033[0m")

    kwargs["video"] = args.video
    kwargs["input_ids"] = input_ids
    kwargs["pixel_values"] = pixel_values
    kwargs["mask"] = mask
    kwargs["temperature"] = args.temperature
    kwargs["max_tokens"] = args.max_tokens

    response = generate(
        model,
        processor,
        prompt=text,
        verbose=args.verbose,
        **kwargs,
    )

    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()
