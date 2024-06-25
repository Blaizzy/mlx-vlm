import mlx.core as mx
import mlx.nn as nn
import numpy as np

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
        if isinstance(module, nn.Linear) and name.split(".")[-1] in linear_layers:
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
        if isinstance(module, cls):
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


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def count_parameters(trainable_params_dict):
    total_params = 0
    for k, v in flatten_dict(trainable_params_dict).items():
        if hasattr(v, "shape"):
            total_params += np.prod(v.shape)

        if isinstance(v, list):
            for v_ in v:
                v_ = flatten_dict(v_)
                if isinstance(v_, dict):
                    total_params += sum(
                        np.prod(p.shape) for p in v_.values() if hasattr(p, "shape")
                    )

    return total_params


def print_trainable_parameters(source_model_trainable, lora_model_trainable):
    lora_trainable_percent = (lora_model_trainable / source_model_trainable) * 100
    print(
        f"#trainable params: {lora_model_trainable} || all params: {source_model_trainable} || trainable%: {lora_trainable_percent}"
    )
