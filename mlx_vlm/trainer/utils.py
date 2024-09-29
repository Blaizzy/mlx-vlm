from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .lora import LoRaLayer


def get_module_by_name(model, name):
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    if parts[-1].isdigit():
        module[int(parts[-1])] = new_module
    else:
        setattr(module, parts[-1], new_module)


def get_peft_model(model, linear_layers, freeze=True, verbose=True):
    source_model_trainable = count_parameters(
        model.language_model.trainable_parameters()
    )

    if freeze:
        freeze_model(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
            if name.split(".")[-1] in linear_layers:
                lora_layer = LoRaLayer(module, 10, 0.1, 0.1)
                set_module_by_name(model, name, lora_layer)

    lora_model_trainable = count_parameters(model.language_model.trainable_parameters())
    if verbose:
        print_trainable_parameters(source_model_trainable, lora_model_trainable)

    return model


def freeze_model(model):
    for name, module in model.named_modules():
        if name in [
            "language_model",
            "vision_model",
            "vision_tower",
            "aligner",
            "connector",
            "multi_modal_projector",
            "mm_projector",
        ]:
            model[f"{name}"].freeze()


def find_all_linear_names(model):
    cls = nn.Linear
    quantized_cls = nn.QuantizedLinear
    lora_module_names = set()
    multimodal_keywords = [
        "mm_projector",
        "vision_tower",
        "vision_resampler",
        "aligner",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) or isinstance(module, quantized_cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def collate_fn(processor, examples):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="np",
        padding="longest",
        tokenize_newline_separately=False,
    )

    tokens = tokens.to(mx.float16)
    return tokens


def count_parameters(trainable_params_dict):
    total_params = 0
    for modules in tree_flatten(trainable_params_dict):
        mx_array = modules[-1]
        if hasattr(mx_array, "shape"):
            total_params += np.prod(mx_array.shape)

    return total_params


def print_trainable_parameters(source_model_trainable, lora_model_trainable):
    lora_trainable_percent = (lora_model_trainable / source_model_trainable) * 100
    print(
        f"#trainable params: {lora_model_trainable} || all params: {source_model_trainable} || trainable%: {lora_trainable_percent}"
    )


def apply_lora_layers(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Apply LoRA layers to the model.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")

    # TODO: add lora params to the config and load them here
    list_of_modules = find_all_linear_names(model.language_model.model)
    model = get_peft_model(model, list_of_modules)

    # TODO: Use custom adapter name
    model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)

    return model
