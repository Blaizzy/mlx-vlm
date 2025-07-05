import copy
import glob
import importlib
import inspect
import json
import logging
import shutil
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
import scipy.signal as signal
import soundfile as sf
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map_with_path, tree_reduce, tree_unflatten
from mlx_lm.utils import quantize_model
from PIL import Image, ImageOps
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .models.base import BaseImageProcessor
from .tokenizer_utils import load_tokenizer
from .trainer import apply_lora_layers

# Constants
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}

MAX_FILE_SIZE_GB = 5

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def skip_multimodal_module(path: str) -> bool:
    """
    Check if a multimodal module (vision/audio) should skip quantization.

    Args:
        path: The module path to check

    Returns:
        bool: True if the module is multimodal and should skip quantization, False otherwise
    """
    return (
        "vision_model" in path
        or "vision_tower" in path
        or "audio_model" in path
        or "audio_tower" in path
    )


def get_model_and_args(config: dict):
    """
    Retrieve the model object based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_vlm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def get_model_path(
    path_or_hf_repo: str, revision: Optional[str] = None, force_download: bool = False
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jinja",
                ],
                force_download=force_download,
            )
        )
    return model_path


def load_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path, **kwargs)
    quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        message = f"""
No safetensors found in {model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id= "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """
        raise FileNotFoundError(message)

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_type = get_model_and_args(config=config)

    # Initialize text and vision configs if not present
    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    # Initialize model config and update it with module configs
    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)

    model = model_class.Model(model_config)

    # Sanitize weights
    weights = sanitize_weights(model, weights)
    weights = sanitize_weights(
        model_class.VisionModel, weights, model_config.vision_config
    )
    weights = sanitize_weights(
        model_class.LanguageModel, weights, model_config.text_config
    )
    if hasattr(model_class, "AudioModel"):
        weights = sanitize_weights(
            model_class.AudioModel, weights, model_config.audio_config
        )

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may or may not have vision quantized
        # TODO: Re-upload the models with the new quantization config and remove this
        skip_vision = config.get("vision_config", {}).get("skip_vision", False)

        def get_class_predicate(p, m):
            # Always skip vision and audio models
            if skip_multimodal_module(p) and skip_vision:
                return False
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            # Skip layers not divisible by 64
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=get_class_predicate,
        )

    model.load_weights(list(weights.items()))
    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def sanitize_weights(model_obj, weights, config=None):
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        if config is not None:
            model_obj = model_obj(config)
        weights = model_obj.sanitize(weights)
    return weights


def update_module_configs(model_config, model_class, config, modules):
    """Updates configuration for model modules like text and vision modules.

    Args:
        model_config: The model configuration object that will be updated
        model_class: The model class containing component config classes
        config: Dictionary containing configuration parameters
        modules: List of module names to update configs for (e.g. ["text", "vision"])

    Returns:
        The updated model_config object
    """
    for config_name in modules:
        config_attr = f"{config_name}_config"
        if hasattr(model_config, config_attr):
            config_class = getattr(model_class, f"{config_name.title()}Config")
            setattr(
                model_config, config_attr, config_class.from_dict(config[config_attr])
            )
    return model_config


def load(
    path_or_hf_repo: str,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    revision: Optional[str] = None,
    **kwargs,
) -> Tuple[nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    force_download = kwargs.get("force_download", False)
    model_path = get_model_path(
        path_or_hf_repo, force_download=force_download, revision=revision
    )
    model = load_model(model_path, lazy, **kwargs)
    if adapter_path is not None:
        model = apply_lora_layers(model, adapter_path)
        model.eval()

    image_processor = load_image_processor(model_path, **kwargs)

    # Get the eos_token_id from the model config
    eos_token_id = getattr(model.config, "eos_token_id", None)

    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)

    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def load_config(model_path: Union[str, Path], **kwargs) -> dict:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments to pass to the config loader

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    try:
        return AutoConfig.from_pretrained(model_path, **kwargs).to_dict()
    except ValueError:
        try:
            with open(model_path / "config.json", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config not found at {model_path}") from exc


def load_image_processor(model_path: Union[str, Path], **kwargs) -> BaseImageProcessor:
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    if not kwargs:
        config = load_config(model_path, trust_remote_code=True)
    else:
        config = load_config(model_path, **kwargs)

    model_class, _ = get_model_and_args(config)
    image_processor = None

    if hasattr(model_class, "ImageProcessor"):
        init_signature = inspect.signature(model_class.ImageProcessor.__init__)

        if "config" in init_signature.parameters:
            image_processor = model_class.ImageProcessor(config=config)
        else:
            image_processor = model_class.ImageProcessor()

    return image_processor


def load_processor(
    model_path, add_detokenizer=True, eos_token_ids=None, **kwargs
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:

    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if add_detokenizer:
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)

        # Get the tokenizer object
        tokenizer_obj = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

        # Instantiate the detokenizer
        processor.detokenizer = detokenizer_class(tokenizer_obj)

        # Determine the EOS token IDs, prioritizing the function argument
        final_eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else tokenizer_obj.eos_token_ids
        )

        # Create and assign the StoppingCriteria
        criteria = StoppingCriteria(final_eos_token_ids, tokenizer_obj)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.stopping_criteria = criteria
        else:
            processor.stopping_criteria = criteria

    return processor


def fetch_from_hub(
    model_path: Path, lazy: bool = False, **kwargs
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy, **kwargs)
    config = load_config(model_path, **kwargs)
    processor = load_processor(
        model_path,
        add_detokenizer=False,
        eos_token_ids=config.get("eos_token_id", None),
        **kwargs,
    )
    return model, config, processor


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = dedent(
        f"""
        # {upload_repo}
        This model was converted to MLX format from [`{hf_path}`]() using mlx-vlm version **{__version__}**.
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        ## Use with mlx

        ```bash
        pip install -U mlx-vlm
        ```

        ```bash
        python -m mlx_vlm.generate --model {upload_repo} --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def save_weights(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    weights = dict(tree_flatten(model.parameters()))
    del model

    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("torch_dtype", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def load_image(image_source: Union[str, Path, BytesIO], timeout: int = 10):
    """
    Helper function to load an image from either a URL or file.
    """
    if isinstance(image_source, BytesIO) or Path(image_source).is_file():
        # for base64 encoded images
        try:
            image = Image.open(image_source)
        except IOError as e:
            raise ValueError(
                f"Failed to load image from {image_source} with error: {e}"
            ) from e
    elif image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True, timeout=timeout)
            response.raise_for_status()
            image = Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            ) from e
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def resize_image(img, max_size):
    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size)


def process_image(img, resize_shape, image_processor):
    if isinstance(img, str):
        img = load_image(img)
    if resize_shape is not None and not isinstance(image_processor, BaseImageProcessor):
        img = resize_image(img, resize_shape)
    return img


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


def load_audio(
    file: str,
    sr: int,
    timeout: int = 10,
):
    """
    Helper function to load audio from either a URL or file.
    """
    if file.startswith(("http://", "https://")):
        try:
            response = requests.get(file, stream=True, timeout=timeout)
            response.raise_for_status()
            audio, sample_rate = sf.read(BytesIO(response.content), always_2d=True)
        except Exception as e:
            raise ValueError(
                f"Failed to load audio from URL: {file} with error {e}"
            ) from e
    else:
        audio, sample_rate = sf.read(file, always_2d=True)

    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return np.array(audio).mean(axis=1)


def process_inputs(
    processor,
    prompts,
    images=None,
    audio=None,
    add_special_tokens=False,
    return_tensors="mlx",
):
    # Get the process method from the processor
    process_method = getattr(processor, "process", processor)

    # Prepare arguments
    args = {
        "text": prompts,
        "images": images,
        "padding": True,
        "return_tensors": return_tensors,
    }

    # Add special tokens if supported
    if "add_special_tokens" in inspect.signature(process_method).parameters:
        args["add_special_tokens"] = add_special_tokens

    # Add audio if provided and supported
    if audio is not None:
        if "audio" in inspect.signature(process_method).parameters:
            args["audio"] = audio
        else:
            raise ValueError(f"Processor {processor} does not support audio parameter")

    return process_method(**args)


def process_inputs_with_fallback(
    processor, prompts, images, audio, add_special_tokens=False, return_tensors="mlx"
):
    # First attempt with specified return_tensors
    try:
        return process_inputs(
            processor,
            prompts=prompts,
            images=images,
            audio=audio,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
    except Exception as e:
        # Fallback to PyTorch tensors if MLX fails
        if return_tensors != "pt":
            try:
                return process_inputs(
                    processor,
                    prompts=prompts,
                    images=images,
                    audio=audio,
                    add_special_tokens=add_special_tokens,
                    return_tensors="pt",
                )
            except Exception as fallback_error:
                raise ValueError(
                    f"Failed to process inputs with error: {fallback_error}"
                )

        raise ValueError(f"Failed to process inputs with error: {e}")


def prepare_inputs(
    processor,
    images=None,
    audio=None,
    prompts=None,
    image_token_index=None,
    resize_shape=None,
    add_special_tokens=False,
):

    if not images and not audio:
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        inputs = tokenizer(prompts, add_special_tokens=add_special_tokens)
        input_ids = mx.array([inputs.input_ids])
        mask = mx.array([inputs.attention_mask])
        return {
            "input_ids": input_ids,
            "attention_mask": mask,
        }

    # Process images
    if images is not None:
        if not isinstance(images, list):
            images = [images]

        image_processor = (
            processor.image_processor if hasattr(processor, "image_processor") else None
        )
        images = [process_image(img, resize_shape, image_processor) for img in images]

    # Process audio
    if audio:
        if not isinstance(audio, list):
            audio = [audio]

        if len(audio) > 1:
            print(
                "\033[33mWarning\033[0m: Single prompt with multiple audio files is not supported yet. Using the first audio file.\n"
            )
            audio = audio[:1]

        audio = [
            load_audio(audio_file, sr=processor.feature_extractor.sampling_rate)
            for audio_file in audio
        ]
    else:
        audio = None

    model_inputs = {}

    if hasattr(processor, "image_processor") and isinstance(
        processor.image_processor, BaseImageProcessor
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]

        processor.pad_token = processor.eos_token
        text_chunks = [
            [processor(chunk).input_ids for chunk in prompt.split("<image>")]
            for prompt in prompts
        ]

        # Find the maximum length for padding
        max_length = max(
            sum(len(chunk) for chunk in chunks) + 1 for chunks in text_chunks
        )

        # Pad and create input_ids
        input_ids = []
        for chunks in text_chunks:
            ids = chunks[0] + [image_token_index] + chunks[1]
            padding = [processor.pad_token_id] * (max_length - len(ids))
            input_ids.append(mx.array(ids + padding))

        model_inputs["input_ids"] = mx.array(input_ids)
        pixel_values = processor.image_processor.preprocess(images=images)
        model_inputs["pixel_values"] = mx.array(np.stack(pixel_values))
        model_inputs["attention_mask"] = mx.array(
            [(ids != processor.pad_token_id) for ids in input_ids]
        ).astype(mx.int32)

    else:
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        inputs = process_inputs_with_fallback(
            processor,
            images=images,
            audio=audio,
            prompts=prompts,
            add_special_tokens=add_special_tokens,
        )

        if "images" in inputs:
            inputs["pixel_values"] = inputs["images"]
            inputs.pop("images")

        model_inputs["attention_mask"] = (
            mx.array(inputs["attention_mask"]) if "attention_mask" in inputs else None
        )
        # Convert inputs to model_inputs with mx.array if present
        for key, value in inputs.items():
            if key not in model_inputs and not isinstance(value, (str, list)):
                model_inputs[key] = mx.array(value)

    return model_inputs


class StoppingCriteria:
    def __init__(self, eos_token_ids: List[int], tokenizer=None):

        if isinstance(eos_token_ids, int):
            self.eos_token_ids = [eos_token_ids]
        else:
            self.eos_token_ids = eos_token_ids

        self.tokenizer = tokenizer

    def add_eos_token_ids(self, new_eos_token_ids: Union[int, List[int]] = None):
        """
        Add new token IDs to the list of EOS token IDs.

        Args:
            new_eos_token_ids: Integer, string, or list of integers/strings representing token IDs to add.
                               If strings are provided, they will be converted to integers if possible.
        """
        if new_eos_token_ids is None:
            return

        if self.tokenizer is None:
            raise ValueError("Processor is not provided")

        if new_eos_token_ids is not None:
            if isinstance(new_eos_token_ids, str):
                new_eos_token_ids = [new_eos_token_ids]
            new_eos_token_ids = [
                self.tokenizer.encode(" " + token, add_special_tokens=False)[-1]
                for token in new_eos_token_ids
            ]
            self.eos_token_ids.extend(new_eos_token_ids)

    def reset(self, eos_token_ids: List[int] = None):
        eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else self.tokenizer.eos_token_ids
        )

        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        if self.eos_token_ids != eos_token_ids:
            self.eos_token_ids = eos_token_ids

    def __call__(self, input_ids: mx.array) -> bool:
        return input_ids in self.eos_token_ids


def print_array_report(t: mx.array, label: Optional[str]) -> dict:
    """
    Return a dictionary report of an MLX array similar to PyTorch's tensor representation.
    Args:
        arr: MLX array to analyze
    Returns:
        Dictionary containing shape, dtype, value representation, and statistics
    """

    from pprint import pprint

    # Get basic statistics
    mean_val = mx.mean(t)
    std_val = mx.std(t)
    min_val = mx.min(t)
    max_val = mx.max(t)

    report = {
        "shape": f"{tuple(t.shape)}",
        "dtype": str(t.dtype),
        "value": repr(t),
        "mean": f"array({mean_val}, dtype={t.dtype})",
        "std": f"array({std_val}, dtype={t.dtype})",
        "min": f"array({min_val}, dtype={t.dtype})",
        "max": f"array({max_val}, dtype={t.dtype})",
        "label": label if label else "array",
    }

    # Print each field, handling 'value' specially
    print("{")
    for key, value in report.items():
        if key == "value":
            print(f" '{key}': {value},")  # No quotes around value
        else:
            print(f" '{key}': {repr(value)},")
    print("}")
    return report
