import copy
import glob
import importlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageOps
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .models.base import BaseImageProcessor, KVCache, SimpleKVCache
from .sample_utils import top_p_sampling
from .tokenizer_utils import load_tokenizer
from .trainer import apply_lora_layers

# Constants
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}

MAX_FILE_SIZE_GB = 5


@dataclass
class GenerationResult:
    text: str
    token: Optional[int]
    logprobs: Optional[List[float]]
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory: float


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


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
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
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
                resume_download=True,
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

    # Get vision config settings with defaults
    vision_config = config.get("vision_config", {})
    skip_vision = vision_config.get("skip_vision", False)

    # Initialize model config and update it with module configs
    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector"]
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

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized`
        class_predicate = get_class_predicate(skip_vision, weights)

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
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


def get_class_predicate(skip_vision, weights=None):
    if skip_vision:
        return lambda p, m: hasattr(m, "to_quantized") and not (
            "vision_model" in p or "vision_tower" in p
        )
    else:
        if weights:
            return lambda p, m: (
                hasattr(m, "to_quantized")
                and m.weight.shape[-1] % 64 == 0
                and f"{p}.scales" in weights
            )
        else:
            return (
                lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0
            )


def load(
    path_or_hf_repo: str,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
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
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, **kwargs)
    if adapter_path is not None:
        # TODO: Support more modules than just language_model
        model = apply_lora_layers(model, adapter_path)
        model.eval()

    image_processor = load_image_processor(model_path, **kwargs)
    processor = load_processor(model_path, True, **kwargs)

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
        import inspect

        init_signature = inspect.signature(model_class.ImageProcessor.__init__)

        if "config" in init_signature.parameters:
            image_processor = model_class.ImageProcessor(config=config)
        else:
            image_processor = model_class.ImageProcessor()

    return image_processor


def load_processor(
    model_path, add_detokenizer=True, **kwargs
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:

    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    if add_detokenizer:
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
        if "tokenizer" in processor.__dict__.keys():
            processor.detokenizer = detokenizer_class(processor.tokenizer)
        else:
            processor.detokenizer = detokenizer_class(processor)
    return processor


def fetch_from_hub(
    model_path: Path, lazy: bool = False, **kwargs
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy, **kwargs)
    config = load_config(model_path, **kwargs)
    processor = load_processor(model_path, add_detokenizer=False, **kwargs)
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


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
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
                    "tokenizer.model",
                    "*.tiktoken",
                    "*.txt",
                ],
            )
        )
    return model_path


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
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
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


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int,
    q_bits: int,
    skip_vision: bool = False,
) -> Tuple[dict, dict]:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.
        skip_vision (bool): Whether to skip quantizing vision model weights.

    Returns:
        Tuple[dict, dict]: Tuple containing quantized weights and updated config.
    """
    quantized_config = copy.deepcopy(config)
    quantized_config.setdefault("vision_config", {})

    # Apply quantization
    if skip_vision:
        # Quantize only non-vision modules
        nn.quantize(
            model,
            q_group_size,
            q_bits,
            class_predicate=get_class_predicate(skip_vision),
        )
        quantized_config["vision_config"]["skip_vision"] = skip_vision

    else:
        # Quantize only layers with to_quantized method and divisible by 64
        nn.quantize(
            model,
            q_group_size,
            q_bits,
            class_predicate=get_class_predicate(skip_vision),
        )

    # Update config and get weights
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def _update_vision_config(
    config: dict,
    value: Union[int, bool],
    divisor: int = 64,
    key: str = "intermediate_size",
) -> None:
    """Update vision config with padded sizes."""
    if key in ["intermediate_size", "hidden_size"]:
        config["vision_config"][key] = (
            ((value // divisor) + 1) * divisor if value % divisor != 0 else value
        )
    else:
        config["vision_config"][key] = value


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

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized linear layers in the model.

    Args:
        model (nn.Module): The model with quantized linear layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    de_quantize_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            bias = "bias" in module
            weight = module.weight
            weight = mx.dequantize(
                weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            output_dims, input_dims = weight.shape
            linear = nn.Linear(input_dims, output_dims, bias=bias)
            linear.weight = weight
            if bias:
                linear.bias = module.bias
            de_quantize_layers.append((name, linear))
    if len(de_quantize_layers) > 0:
        model.update_modules(tree_unflatten(de_quantize_layers))
    return model


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    skip_vision: bool = False,
    trust_remote_code: bool = True,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=trust_remote_code
    )

    weights = dict(tree_flatten(model.parameters()))
    dtype = getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, skip_vision
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    processor.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)


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


def prepare_inputs(processor, images, prompts, image_token_index, resize_shape=None):

    if not isinstance(images, list):
        images = [images]

    # Process images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )
    images = [process_image(img, resize_shape, image_processor) for img in images]

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

        if hasattr(processor, "process"):
            inputs = processor.process(
                text=prompts, images=images, padding=True, return_tensors="mlx"
            )
        else:
            inputs = processor(
                text=prompts, images=images, padding=True, return_tensors="mlx"
            )

        if "images" in inputs:
            inputs["pixel_values"] = inputs["images"]
            inputs.pop("images")

        if isinstance(inputs["pixel_values"], list):
            pixel_values = inputs["pixel_values"]
        else:
            pixel_values = mx.array(inputs["pixel_values"])

        model_inputs["pixel_values"] = pixel_values
        model_inputs["attention_mask"] = (
            mx.array(inputs["attention_mask"]) if "attention_mask" in inputs else None
        )
        # Convert inputs to model_inputs with mx.array if present
        for key, value in inputs.items():
            if key not in model_inputs and not isinstance(value, (str, list)):
                model_inputs[key] = mx.array(value)

    return model_inputs


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        logit_bias (dictionary, optional): Additive logit bias.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        logprobs = logits - mx.logsumexp(logits)

        if temperature == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temperature)
            else:
                token = mx.random.categorical(logits * (1 / temperature))

        return token, logprobs

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = input_ids
    if hasattr(model.language_model, "make_cache"):
        cache = model.language_model.make_cache()
    else:
        kv_heads = (
            [model.language_model.n_kv_heads] * len(model.language_model.layers)
            if isinstance(model.language_model.n_kv_heads, int)
            else model.language_model.n_kv_heads
        )
        if model.config.model_type == "florence2":
            cache = [
                (SimpleKVCache(), SimpleKVCache()) for n in model.language_model.layers
            ]
        else:
            cache = [KVCache(model.language_model.head_dim, n) for n in kv_heads]

    repetition_context = input_ids.reshape(-1).tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    def _step(y, **kwargs):
        nonlocal repetition_context
        if "decoder_input_ids" in kwargs:
            outputs = model.language_model(
                cache=cache,
                **kwargs,
            )
        else:

            outputs = model.language_model(
                y[None],
                cache=cache,
                mask=mask,
                **kwargs,
            )

        logits = outputs.logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, logprobs = sample(logits)
            repetition_context.append(y.item())
        else:
            y, logprobs = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        return y, logprobs.squeeze(0)

    outputs = model(input_ids, pixel_values, cache=cache, mask=mask, **kwargs)

    logits = outputs.logits[:, -1, :]
    y, logprobs = sample(logits)
    mx.async_eval(y)

    if outputs.cross_attention_states is not None:
        kwargs = {
            k: v
            for k, v in zip(
                ["cross_attention_states"], [outputs.cross_attention_states]
            )
        }
    elif outputs.encoder_outputs is not None:
        kwargs = {
            "decoder_input_ids": y[None],
            "encoder_outputs": outputs.encoder_outputs,
        }
    else:
        kwargs = {}

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y, **kwargs)
            mx.async_eval(next_y)
            if "decoder_input_ids" in kwargs:
                kwargs["decoder_input_ids"] = next_y[None]
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs
        if n == max_tokens:
            break

        n += 1


def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The ma
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing text.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)

    if kwargs.get("pixel_values") is None:
        if not image:
            input_ids = prompt_tokens[None, :]
            pixel_values = mask = None
        else:
            inputs = prepare_inputs(
                processor, image, prompt, image_token_index, resize_shape
            )
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            mask = inputs["attention_mask"]
            data_kwargs = {
                k: v
                for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            }
            kwargs.update(data_kwargs)
    else:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")

    detokenizer = processor.detokenizer
    detokenizer.reset()
    tic = time.perf_counter()
    for n, (token, logprobs) in enumerate(
        generate_step(input_ids, model, pixel_values, mask, **kwargs)
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            prompt_tps = input_ids.size / prompt_time
            tic = time.perf_counter()

        if token == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token)

        # Yield the last segment if streaming
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=input_ids.size,
            generation_tokens=n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.metal.get_peak_memory() / 1e9,
        )

    detokenizer.finalize()
    yield GenerationResult(
        text=detokenizer.last_segment,
        token=token,
        logprobs=logprobs,
        prompt_tokens=input_ids.size,
        generation_tokens=n + 1,
        prompt_tps=prompt_tps,
        generation_tps=(n + 1) / (time.perf_counter() - tic),
        peak_memory=mx.metal.get_peak_memory() / 1e9,
    )


def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        if image is not None:
            input_path = image
        elif kwargs.get("video") is not None:
            input_path = kwargs.get("video")
        else:
            input_path = None

        print(f"Files: {input_path}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    for response in stream_generate(model, processor, prompt, image, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return text
