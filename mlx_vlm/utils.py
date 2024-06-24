import copy
import glob
import importlib
import json
import logging
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
from .tokenizer_utils import load_tokenizer

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


def load_model(model_path: Path, lazy: bool = False) -> nn.Module:
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

    if "language_config" in config:
        config["text_config"] = config["language_config"]
        del config["language_config"]

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
    if model_type == "phi3_v":
        config["vision_config"] = config["img_processor"]
        config["text_config"] = {}

    model_config = model_class.ModelConfig.from_dict(config)

    model_config.vision_config = model_class.VisionConfig.from_dict(
        config["vision_config"]
    )

    model_config.text_config = model_class.TextConfig.from_dict(config["text_config"])

    if hasattr(model_config, "perceiver_config"):
        model_config.perceiver_config = model_class.PerceiverConfig.from_dict(
            config["perceiver_config"]
        )
    if hasattr(model_config, "aligner_config"):
        model_config.aligner_config = model_class.AlignerConfig.from_dict(
            config["aligner_config"]
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

    model = load_model(model_path, lazy)
    processor = load_processor(model_path, processor_config=processor_config)

    return model, processor


def load_config(model_path: Union[str, Path]) -> dict:

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
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    config = load_config(model_path)
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
    model_path, processor_config={"trust_remote_code": True}
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
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
    model = load_model(model_path, lazy)
    config = load_config(model_path)
    processor = load_processor(model_path)

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
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        Tuple: Tuple containing quantized weights and config.
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

    if "vision_config" in quantized_config:
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
):
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
    Helper function to load an image from either a URL or file.
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
        if "image_sizes" in inputs:
            return input_ids, pixel_values, inputs["image_sizes"]
    return input_ids, pixel_values, mask


def sample(logits: mx.array, temp: float, top_p: float) -> Tuple[mx.array, float]:
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
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty (default 20).
        top_p (float, optional): Nulceus sampling, higher means model considers more less likely words

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
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
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
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
