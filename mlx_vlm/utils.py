"""

Module Document: `utils`

This module provides a wide array of functionalities to support models from MLX (Model Library eXchange). Its primary focus is on downloading models from the Hugging Face Hub, loading and processing models, handling tokenization, and assisting with input generation, quantization, dequantization, and serialization of models.

Key components of the module include:

- Functionality to fetch model configurations and load models along with their weights.
- Support for image processors to handle different image input types and preprocessing before model inference.
- Utilities for handling tokenizers, including loading and wrapping existing tokenizer implementations.
- Functions to assist with model quantization and dequantization for efficient storage and computation.
- Functions to prepare inputs for models, which handle the construction of appropriate inputs from text prompts and images.
- Sampling utility functions to aid with temperature-controlled and top-p sampling from model output distributions.
- Helper functions to manage model serialization, including the creation of weight shards to manage large models within filesystem limitations.
- Utility functions to upload serialized models to the Hugging Face Hub.

The module ensures that the models from MLX are easily accessible, customizable, and efficient for use in different environments. Its design provides a seamless bridge between MLX's model capabilities and the extensive resources available through the Hugging Face Hub, enabling developers and data scientists to incorporate advanced ML models into their applications with ease.
"""

import copy
import glob
import importlib
import json
import logging
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .models.base import BaseImageProcessor
from .sample_utils import top_p_sampling
from .tokenizer_utils import TokenizerWrapper, load_tokenizer

# Constants
MODEL_REMAPPING = {
    "llava-qwen2": "nanoLlava",
}

MAX_FILE_SIZE_GB = 5

linear_class_predicate = (
    lambda m: isinstance(m, nn.Linear)
    and m.weight.shape[0]
    != 8  # avoid quantizing gate layers, otherwise we have to re-quant and upload all the mixtral models
)


def get_model_and_args(config: dict):
    """
    Generates and returns a message structure for a given model and prompt.
    This function takes a model name and a prompt, then constructs a message object in the appropriate
    format expected by the model. The output is a json-compatible dictionary that contains keys like
    'role' and 'content', which in turn may contain keys like 'type' and 'text'. The format of the message
    depends on the model's specific requirements.

    Args:
        model_name (str):
             The name of the model for which the message is being generated.
        prompt (str):
             The input or instruction for the model to process.

    Returns:
        (dict):
             A structured message in the form of a dictionary compatible with the model's API.

    Raises:
        ValueError:
             If the 'model_name' is not supported, a ValueError is raised with an appropriate
            error message detailing the unsupported model.

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
    Gets the path to a machine learning model, either from a local path or from a HuggingFace repository.
    This function helps in locating and optionally downloading the model specified by path_or_hf_repo and
    its associated files. If the given path exists locally, it is simply returned. If the path does not exist,
    an attempt is made to download the model from the HuggingFace Hub based on the repository identifier.
    The function allows specific patterns of files to be downloaded and supports the resume of downloads.

    Args:
        path_or_hf_repo (str):
             A string representing a local filesystem path or a HuggingFace repository identifier.
        revision (Optional[str], optional):
              The specific model revision to download. Defaults to None.

    Returns:
        (Path):
             A pathlib.Path object representing the path to the downloaded or local model.

    Raises:
        OSError:
             If the model files cannot be located or downloaded from the specified path or repository.

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


def load_model(model_path: Path, lazy: bool = False) -> nn.Module:
    """
    Loads a model from a specified path containing model configurations and weights.
    This function first calls `load_config` to load the model configuration from a JSON file. It then looks for safetensor files (weights) in the model's directory. If no weights are found, it raises a `FileNotFoundError` with a detailed error message instructing how to generate safetensors from a Hugging Face model.
    The function retrieves the model and its arguments based on the configuration file, catering specifically to different model types by adjusting the configurations accordingly. It instantiates the model using the configuration class provided by `model_class.ModelConfig` and loads the weights.
    If the model's configuration specifies quantization settings, the function quantizes the model before it is fully loaded.
    The function also ensures that the subset of the model's parameters designated as 'sanitized' is properly formatted before loading. After loading the weights, it sets the model to evaluation mode.
    In cases where the `lazy` argument is `False`, the function also initialises the model parameters for evaluation.

    Args:
        model_path (Path):
             A pathlib `Path` object pointing to the directory where the model configuration and safetensors (weights) are stored.
        lazy (bool, optional):
             If set to `True`, the model parameters will not be initialised for evaluation. Defaults to `False`.

    Returns:
        (nn.Module):
             The fully loaded and instantiated model ready for evaluation.

    Raises:
        FileNotFoundError:
             If no safetensors are found in the provided `model_path` or if configuration files are missing.
        ValueError:
             If the configuration includes settings for an unsupported model type.

    """

    config = load_config(model_path)
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

    if model_type == "nanoLlava":
        vision_config = AutoConfig.from_pretrained(config["mm_vision_tower"])
        text_config = AutoConfig.from_pretrained(config["language_model"])
        vision_config = vision_config.to_dict()
        text_config = text_config.to_dict()
        config["vision_config"] = vision_config["vision_config"]
        config["text_config"] = text_config
    if model_type == "idefics2":
        config = AutoConfig.from_pretrained(model_path).to_dict()

    model_config = model_class.ModelConfig.from_dict(config)

    model_config.vision_config = model_class.VisionConfig.from_dict(
        config["vision_config"]
    )

    model_config.text_config = model_class.TextConfig.from_dict(config["text_config"])

    if hasattr(model_config, "perceiver_config"):
        model_config.perceiver_config = model_class.PerceiverConfig.from_dict(
            config["perceiver_config"]
        )
    model = model_class.Model(model_config)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if hasattr(model_class.VisionModel, "sanitize"):
        weights = model_class.VisionModel(model_config.vision_config).sanitize(
            weights=weights
        )

    if hasattr(model_class.LanguageModel, "sanitize"):
        weights = model_class.LanguageModel(model_config.text_config).sanitize(
            weights=weights
        )

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
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


def load(
    path_or_hf_repo: str,
    processor_config={},
    lazy: bool = False,
) -> Tuple[nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Loads a model and processor given a file path or Hugging Face repository name.
    This function is designed to simplify the process of loading a model and its associated
    processor using either a local file path or the name of a model repository from Hugging Face.
    It leverages an internal get_model_path function to retrieve the model path and then calls
    the load_model and load_processor functions to load the model and processor respectively.

    Args:
        path_or_hf_repo (str):
             A string representing either the local file path to the model
            directory or the exact name of the Hugging Face model repository.
        processor_config (dict, optional):
             A dictionary of configuration options for the processor.
            Defaults to an empty dictionary.
        lazy (bool, optional):
             A flag to determine whether or not to lazily initialize the model.
            If True, the model is initialized but not fully loaded until necessary. This can be
            beneficial when loading large models to save memory. Defaults to False.

    Returns:
        (Tuple[nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]):
             A tuple containing
            the loaded model and processor. The model is an instance of nn.Module, and the processor
            is either an instance of PreTrainedTokenizer or PreTrainedTokenizerFast depending on the
            model that was loaded.

    Raises:
        FileNotFoundError:
             If the model or processor files cannot be found at the specified path or
            repository name.
        ValueError:
             If an unsupported model type is encountered during the loading process.

    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy)
    processor = load_processor(model_path, processor_config=processor_config)

    return model, processor


def load_config(model_path: Union[str, Path]) -> dict:
    """
    Loads a configuration file from a given model path.

    Args:
        model_path (Union[str, Path]):
             A string or Path object representing the
            path to the model directory. If a string is provided, it will be
            converted to a Path object and validated.

    Returns:
        (dict):
             A dictionary containing configuration parameters.

    Raises:
        FileNotFoundError:
             If the 'config.json' file is not found in the specified
            model path directory.
        TypeError:
             If `model_path` input is neither a string nor a Path object.

    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_image_processor(model_path: Union[str, Path]) -> BaseImageProcessor:
    """
    Loads an image processor for a given model path.
    The function accepts a model path as either a string or a Path object. It attempts to resolve
    the model path and load the associated configuration file. It then retrieves the model class
    based on the model type specified in the configuration and checks if the model class has an
    attribute 'ImageProcessor'. If so, it initializes and returns an instance of the ImageProcessor
    class associated with the model. If the model path does not exist or the model class does not
    have an ImageProcessor attribute, the function returns None.

    Args:
        model_path (Union[str, Path]):
             The file path or Hugging Face repository name where the model
            and associated files are located.

    Returns:
        (BaseImageProcessor):
             An instance of the ImageProcessor class if available, otherwise None.

    Raises:
        FileNotFoundError:
             If the config.json file cannot be found within the resolved model path.
        ValueError:
             If the specified model type is not supported (not present in MODEL_REMAPPING or
            the mlx_vlm.models package).
        ImportError:
             If the module containing the model architecture cannot be imported.

    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    config = load_config(model_path)
    model_class, _ = get_model_and_args(config)
    image_processor = None

    if hasattr(model_class, "ImageProcessor"):
        image_processor = model_class.ImageProcessor()

    return image_processor


def load_processor(
    model_path, processor_config={"trust_remote_code": True}
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Loads a processor for a specific model, given a model path and an optional processor configuration.
    This function initializes an `AutoProcessor` from a pretrained model, which may include both a tokenizer and a feature extractor, depending on the model. Additionally, it identifies an appropriate detokenizer class to instantiate a detokenizer compatible with the loaded tokenizer. The function attaches this detokenizer instance to the processor object before returning it.

    Args:
        model_path (str | os.PathLike):
             The path to the directory where the pretrained model is saved.
        processor_config (dict, optional):
             Additional configuration parameters for initializing the processor. Defaults to {'trust_remote_code': True}.

    Returns:
        (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
             An instance of a processor class that includes both the tokenizer and detokenizer components.

    Raises:
        Any exceptions raised during the instantiation of `AutoProcessor` or related components are propagated without modification.

    """
    processor = AutoProcessor.from_pretrained(model_path, **processor_config)
    detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
    if "tokenizer" in processor.__dict__.keys():
        processor.detokenizer = detokenizer_class(processor.tokenizer)
    else:
        processor.detokenizer = detokenizer_class(processor)
    return processor


def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    """
    Fetches a model, its configuration, and processor from the hub based on a specified path.
    This function retrieves a model, its configuration dictionary, and tokenizer/processor from a remote source or
    local cache. It's meant to be used when working with models hosted on HuggingFace's model hub or similar platforms.

    Args:
        model_path (Path):
             The path to the model directory or the identifier of the hosted model repository.
        lazy (bool, optional):
             A flag that indicates whether to load the model weights immediately or do it
            in a lazy manner (i.e., weights are loaded when needed). Defaults to False.

    Returns:
        (Tuple[nn.Module, dict, PreTrainedTokenizer]):
             A tuple containing the loaded model (as an instance of nn.Module),
            the configuration dictionary, and the pretrained tokenizer/processor.

    Raises:
        FileNotFoundError:
             If any expected files, such as the configuration file or model weights, are not found in
            the provided model_path.
        ValueError:
             If the model type specified in the configuration is not supported.

    """
    model = load_model(model_path, lazy)
    config = load_config(model_path)
    processor = load_processor(model_path)

    return model, config, processor


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Generates a list of shards from a dictionary of weights, where each shard is a sub-dictionary of weights that does not exceed a specified maximum file size in gigabytes.

    Args:
        weights (dict):
             A dictionary with keys representing weight names and values as numerical objects that have the attribute `nbytes`.
        max_file_size_gb (int, optional):
             The maximum file size for each shard in gigabytes. Default value is defined by the constant `MAX_FILE_FILE_SIZE_GB`.

    Returns:
        (list):
             A list of shard dictionaries. Each shard is a sub-dictionary of weights where the total size in bytes does not exceed the `max_file_size_bytes` derived from the `max_file_size_gb`.

    Raises:
        ValueError:
             If a single weight size in bytes is larger than max_file_size_bytes, making it impossible to create a valid shard.

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
    Uploads a local directory to the Hugging Face Model Hub under a specified repository name.

    Args:
        path (str):
             The local directory path which contains the files to be uploaded.
        upload_repo (str):
             The name of the repository on the Hugging Face Model Hub where the files will be uploaded.
        hf_path (str):
             The path to the original model card on the Hugging Face Model Hub, used to reference in the newly uploaded README.

    Raises:
        Exception:
             If any issue occurs during the creation of the repository, the upload of files, or saving of the model card.

    Note:
        This function assumes that the user is already logged into the Hugging Face Hub using their api key. If not, the function will fail and prompt the user to log in.

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
        python -m mlx_vlm.generate --model {upload_repo} --max-tokens 100 --temp 0.0
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
    Fetches the path to a model, downloading it if necessary.
    Retrieves the path to a model specified by either a local file path or a Hugging Face repository.
    If the model does not exist locally at the given path, it will attempt to download it from the provided Hugging Face repository.
    Only files matching certain patterns (such as '*.json', '*.safetensors', '*.py', 'tokenizer.model', '*.tiktoken', '*.txt')
    will be downloaded.
    If a specific revision of the model is required, it can be specified with the 'revision' parameter.
    If 'revision' is not provided, the default branch of the repository will be used.
    The function will return the path to the local model directory.

    Parameters:
        path_or_hf_repo (str):
             A string representing either the local file path or the Hugging Face repository ID of the model.
        revision (Optional[str], optional):
             The specific revision of the model to use. Defaults to None, which will use the default branch.

    Returns:
        (Path):
             A Path object representing the local path to the model.

    Raises:
        ModelDownloadError:
             An exception indicating that the download failed or the model could not be found at the specified location.

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
    Applies a repetition penalty to logits based on previously generated tokens.

    Args:
        logits (mx.array):
             The logits array of shape (batch_size, vocabulary_size). Logits represent
            the unnormalized prediction scores for each vocabulary item.
        generated_tokens (Any):
             The sequence of generated tokens. This should be convertible
            to an array format that MXNet operations can handle. The specific type or collection
            is not prescribed to allow for flexibility in handling different token representations.
        penalty (float):
             The penalty factor to apply to the logits of previously generated tokens.
            Values greater than 1.0 decrease the likelihood of previous tokens being selected again,
            while values less than 1.0 increase their likelihood.

    Returns:
        (mx.array):
             The updated logits array with penalties applied to the scores of previously
            generated tokens.

    Raises:
        TypeError:
             If the `generated_tokens` cannot be converted to a format suitable for MXNet operations.

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
    """
    Saves the provided weights to a specified path in `safetensors` format, with consideration for file size restrictions and optional weight donation behavior.
    This function saves the given weights as multiple files if their combined size exceeds the maximum file size for a single file.
    It organizes these files into shards, where each shard is a group of weights that does not surpass the maximum allowed file size.
    The weights are saved in a directory specified by `save_path`, and the directory is created if it doesn't exist.
    If `donate_weights` is set to True, the original weights dictionary is cleared and its memory is deallocated after saving.

    Args:
        save_path (Union[str, Path]):
             The path (as a string or `pathlib.Path`) to the directory where the weight files are saved.
        weights (Dict[str, Any]):
             A dictionary where keys are weight names and values are the weight values.
        donate_weights (bool, optional):
             If set to True, the original weights dictionary is cleared and its memory is deallocated. Defaults to False.

    Raises:
        FileNotFoundError:
             If `save_path` is not found and cannot be created.

    Returns:
        (None):
             The function does not return any values and operates solely via side effects, such as writing files to disk.

    """
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
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    """
    Quantizes the given neural network model according to the specified configuration, group size, and bit precision.
    The function modifies the intermediate layers of a vision model if their dimensions are not divisible by a
    particular divisor (default 64). It pads the dimensions to make them divisible, ensuring the proper functionality
    of grouped quantization. After adjusting the dimensions, it quantizes the model's weights using the specified
    number of bits and group size, which are aspects of the quantization process that affect the model's
    precision and performance.

    Args:
        model (nn.Module):
             The neural network model to be quantized.
        config (dict):
             A dictionary containing the model's configuration parameters.
        q_group_size (int):
             The size of the weight groups to be used in quantization.
        q_bits (int):
             The number of bits to use for quantizing the weights.

    Returns:
        (Tuple):
             A tuple containing the quantized weights of the model and the new quantized configuration.

    Raises:
        ValueError:
             If the input configuration or model is inappropriate for the quantization process.

    """
    quantized_config = copy.deepcopy(config)
    vision_intermediate_size = model.config.vision_config.intermediate_size
    divisor = 64
    if any(vision_intermediate_size % size != 0 for size in [64, 128]):
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)
                and ("vision_model" in name or "vision_tower" in name)
            ):
                out_features, in_features = module.weight.shape

                # Calculate the padding needed for each dimension
                new_out_features = (
                    ((out_features // divisor) + 1) * divisor
                    if out_features % divisor != 0
                    else out_features
                )
                new_in_features = (
                    ((in_features // divisor) + 1) * divisor
                    if in_features % divisor != 0
                    else in_features
                )
                if (
                    out_features == vision_intermediate_size
                    or in_features == vision_intermediate_size
                ):

                    # If padding is needed, proceed
                    if (
                        new_out_features != out_features
                        or new_in_features != in_features
                    ):
                        # Create new weight and bias tensors
                        new_weight = mx.zeros((new_out_features, new_in_features))
                        new_bias = mx.zeros((new_out_features))

                        # Copy existing weights and biases to the new tensors
                        new_weight[:out_features, :in_features] = module.weight
                        module.weight = new_weight

                        if hasattr(module, "bias"):
                            new_bias[:out_features] = module.bias
                            module.bias = new_bias

    quantized_config["vision_config"]["intermediate_size"] = (
        ((vision_intermediate_size // divisor) + 1) * divisor
        if vision_intermediate_size % divisor != 0
        else vision_intermediate_size
    )

    nn.quantize(model, q_group_size, q_bits)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """
    Saves a configuration dictionary to a specified file after removing a specific key and sorting the dictionary.

    Args:
        config (dict):
             Configuration dictionary that needs to be saved. The function
            removes the key '_name_or_path' from it if present before saving.
        config_path (Union[str, Path]):
             The file path where the configuration should
            be saved. This can be a string or a Path object.

    Raises:
        IOError:
             If there is an issue opening or writing to the file specified by
            config_path, an IOError is raised.

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
    Dequantizes a given neural network model by converting any quantized linear layers to standard linear layers with floating-point precision.

    Args:
        model (nn.Module):
             The neural network model containing the layers to be dequantized.

    Returns:
        (nn.Module):
             The dequantized neural network model with standard linear layers.

    Raises:
        ValueError:
             If the dequantization process encounters unsupported layer configurations or data types.

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
):
    """
    Converts a model from Hugging Face's hub to MLX format with optional quantization or dequantization, saves the converted model locally or uploads it to the Hugging Face hub.

    Args:
        hf_path (str):
             The Hugging Face repository identifier or local path of the original model.
        mlx_path (str):
             The local path where the converted MLX model will be saved. Defaults to 'mlx_model'.
        quantize (bool):
             Whether to quantize the model weights. Defaults to False.
        q_group_size (int):
             Group size used in quantization. Defaults to 64.
        q_bits (int):
             Number of bits used in quantization. Defaults to 4.
        dtype (str):
             Type to cast the model parameters before saving. Defaults to 'float16'.
        upload_repo (str):
             Hugging Face repository name where the converted model will be uploaded. If None, the model will not be uploaded. Defaults to None.
        revision (Optional[str]):
             The specific model revision to convert. Defaults to None.
        dequantize (bool):
             Whether to dequantize the model weights. Defaults to False.

    Raises:
        ValueError:
             If both quantize and dequantize arguments are True.

    Returns:
        (None):
             The function does not return any value.

    """
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=False)

    weights = dict(tree_flatten(model.parameters()))
    dtype = mx.float16 if quantize else getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, q_group_size, q_bits)

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

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)


def load_image(image_source: Union[str, Path, BytesIO]):
    """
    Loads an image from a given source which can be a URL, a file path, or a BytesIO stream.

    Args:
        image_source (Union[str, Path, BytesIO]):
             The source of the image to load. This can be a URL (str),
            a file system path (str or Path), or an in-memory BytesIO stream.

    Returns:
        (PIL.Image.Image):
             The loaded image as a PIL image object.

    Raises:
        ValueError:
             If the image_source is not a valid URL, file path, or BytesIO stream;
            if the image cannot be loaded from the provided BytesIO stream;
            if the image cannot be fetched from the provided URL due to either network
            issues or the server's response;
            if the image cannot be opened from the given file path due to the file not
            existing or being unreadable.

    """
    if isinstance(image_source, BytesIO):
        # for base64 encoded images
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image from BytesIO with error: {e}")
    elif image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except Exception as e:
            raise ValueError(
                f"Failed to load image from URL: {image_source} with error {e}"
            )
    elif Path(image_source).is_file():
        try:
            return Image.open(image_source)
        except IOError as e:
            raise ValueError(f"Failed to load image {image_source} with error: {e}")
    else:
        raise ValueError(
            f"The image {image_source} must be a valid URL or existing file."
        )


def prepare_inputs(image_processor, processor, image, prompt, image_token_index):
    """
    Prepares the inputs required for image and text processing for a given image and prompt.
    This function supports loading and processing an image, splitting a prompt with specialized
    placeholders into chunks, and preparing image and text tensors for input to a model.
    If an `image_processor` is provided, it gets used to preprocess the image. Otherwise, processing
    is done using the `processor`. The image is loaded either from a provided URL or from the local
    filesystem. Attention masks are also generated if necessary.

    Args:
        image_processor (BaseImageProcessor):
             An instance of an image processor to preprocess
            the image.
        processor (transformers.PreTrainedTokenizer):
             An instance of a tokenizer to process the
            textual input.
        image (Image.Image or str):
             The image to be processed or a URL/path pointing to the image.
        prompt (str):
             The text prompt which may include placeholders for the image.
        image_token_index (int):
             The index of the image token in the tokenizer's vocabulary.

    Returns:
        (Tuple[ndarray, ndarray, Optional[ndarray]]):
             A tuple containing the input_ids, pixel_values,
            and an optional attention_mask. If `image_processor` is not provided, attention_mask will
            not be returned.

    Raises:
        ValueError:
             If the provided `image` argument is a string that neither corresponds to a URL
            nor a file path, or if it fails to load as an Image object.
        Imports:
            from mxnet import ndarray as mx
            from transformers.image_utils import load_image
            import numpy as np
            import io

    """
    from transformers.image_utils import load_image

    mask = None
    if isinstance(image, str):
        image = load_image(image)

    if image_processor is not None:
        text_chunks = [processor(chunk).input_ids for chunk in prompt.split("<image>")]
        input_ids = mx.array([text_chunks[0] + [image_token_index] + text_chunks[1]])
        pixel_values = image_processor.preprocess(images=[image])[0]
        pixel_values = mx.array(np.expand_dims(pixel_values, axis=0))
    else:
        inputs = processor(prompt, image, return_tensors="np")
        pixel_values = mx.array(inputs["pixel_values"])
        input_ids = mx.array(inputs["input_ids"])
        mask = mx.array(inputs["attention_mask"])
    return input_ids, pixel_values, mask


def sample(logits: mx.array, temp: float, top_p: float) -> Tuple[mx.array, float]:
    """
    Generates a single token sample from the provided logits using specified sampling techniques.

    Args:
        logits (mx.array):
             The array of logits from which to sample.
        temp (float):
             The temperature to use for scaling the logits before sampling.
            A temperature closer to 0 makes the model more deterministic, with 0 acting as a greedy
            selection from the highest logit. Higher temperatures result in more diversity.
        top_p (float):
             The nucleus sampling probability threshold. If set between 0 and 1,
            it performs top-p (nucleus) sampling, which selects the smallest set of tokens whose
            cumulative probability exceeds the threshold of top_p. If set to a value outside this range,
            top-p sampling is not applied.

    Returns:
        (Tuple[mx.array, float]):
             A tuple where the first element is the sampled token as an mx.array
            and the second element is the corresponding probability of the selected token.

    Raises:
        ValueError:
             If 'top_p' is in the range (0, 1) but 'top_p_sampling' is defined to return
            an unsupported type or if any other precondition for 'top_p_sampling' is violated.

    Note:
        The returned probability corresponds to the softmax probability of the chosen token
        after applying the temperature scaling.

    """
    softmax_logits = mx.softmax(logits)

    if temp == 0:
        token = mx.argmax(logits, axis=-1)
    else:
        if top_p > 0 and top_p < 1.0:
            token = top_p_sampling(logits, top_p, temp)
        else:
            token = mx.random.categorical(logits * (1 / temp))

    prob = softmax_logits[0, token]
    return token, prob


def generate_step(
    model: nn.Module,
    prompt: mx.array,
    mask: mx.array,
    cache=None,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    Generates a single step for a language generation model given a prompt and various generation parameters.
    This function is a generator that yields a tuple containing the generated token and its associated probability. It allows for incorporating custom stopping criteria or other logic in the generation process managed by the caller.

    Args:
        model (nn.Module):
             The language generation model to use for generating the next token.
        prompt (mx.array):
             The input prompt tokens as an MXNet array to the model.
        mask (mx.array):
             The mask array that indicates which tokens are part of the prompt and which are generated.
        cache (Optional, default=None):
             A cache that stores information from previous steps, which can speed up subsequent token generation.
        temp (float, default=0.0):
             The temperature to use for sampling. A lower temperature tends to produce more deterministic output.
        repetition_penalty (Optional[float], default=None):
             The penalty to apply to tokens that already appeared in the context. A higher penalty discourages repetition.
        repetition_context_size (Optional[int], default=20):
             The number of previously generated tokens to consider when applying the repetition penalty.
        top_p (float, default=1.0):
             The nucleous sampling parameter that controls the size of the probability distribution from which to sample. Lower values lead to more focused generation.

    Yields:
        Tuple[mx.array, mx.array]:
             A tuple containing the generated token and its associated probability.

    Raises:
        ValueError:
             If the repetition_penalty is not a non-negative float.

    """

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y, prob = sample(prompt, temp, top_p)

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    while True:
        logits, cache = model(y[None], mask=mask, cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, prob = sample(logits, temp, top_p)
            repetition_context.append(y.item())
        else:
            y, prob = sample(logits, temp, top_p)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        yield y, prob


def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    image: str,
    prompt: str,
    image_processor=None,
    temp: float = 0.0,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
) -> str:
    """
    Generates a text response based on a given image and textual prompt using a provided model and processor.
    This function pre-processes the input image and prompt, fetches the corresponding logits from the model, and
    samples the continuation tokens based on the provided parameters such as temperature, top_p, etc. It supports
    additional options like repetition penalty and verbose output.

    Args:
        model (nn.Module):
             The model to use for generating the output text.
        processor (PreTrainedTokenizer):
             Tokenizer to encode the text prompt and process the generated tokens.
        image (str):
             An image file path or a URL to use as a part of the input.
        prompt (str):
             The text prompt to use as a starting point for text generation.
        image_processor (optional):
             The processor to use for image pre-processing if required by the model.
        temp (float, optional):
             The sampling temperature to use when selecting tokens. Defaults to 0.0.
        max_tokens (int, optional):
             The maximum number of tokens to generate. Defaults to 100.
        verbose (bool, optional):
             Whether to print out verbose information during generation. Defaults to False.
        formatter (Callable, optional):
             A formatter function for verbose output of generated tokens.
        repetition_penalty (float, optional):
             The penalty applied to discourage repetition.
        repetition_context_size (int, optional):
             The size of the context window for applying repetition penalty.
        top_p (float, optional):
             The nucleus sampling hyperparameter controlling the size of the probability mass
            considered for sampling tokens. Defaults to 1.0.

    Returns:
        (str):
             The generated text based on the input image and prompt.

    """
    if verbose:
        print("=" * 10)
        print("Image:", image, "\n")
        print("Prompt:", prompt)

    if image_processor is not None:
        prompt_tokens = mx.array(processor.encode(prompt))
        tokenizer = processor
    else:
        prompt_tokens = mx.array(processor.tokenizer.encode(prompt))
        tokenizer = processor.tokenizer

    image_token_index = model.config.image_token_index
    input_ids, pixel_values, mask = prepare_inputs(
        image_processor, processor, image, prompt, image_token_index
    )
    logits, cache = model(input_ids, pixel_values, mask)
    logits = logits[:, -1, :]
    y, _ = sample(logits, temp, top_p)

    tic = time.perf_counter()
    detokenizer = processor.detokenizer
    detokenizer.reset()

    detokenizer.add_token(y.item())

    for (token, prob), n in zip(
        generate_step(
            model.language_model,
            logits,
            mask,
            cache,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
        ),
        range(max_tokens),
    ):
        token = token.item()
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()

        if token == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token)

        if verbose:
            if formatter:
                # We have to finalize so that the prob corresponds to the last segment
                detokenizer.finalize()
                formatter(detokenizer.last_segment, prob.item())
            else:
                print(detokenizer.last_segment, end="", flush=True)

    token_count = n + 1
    detokenizer.finalize()

    if verbose:
        print(detokenizer.last_segment, flush=True)
        gen_time = time.perf_counter() - tic
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time

        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    return detokenizer.text
