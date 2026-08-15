import glob
import json
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .utils import (
    _load_safetensors,
    apply_generation_config_defaults,
    get_model_and_args,
    load_config,
    sanitize_weights,
    update_module_configs,
)


def _weight_files(model_path: Path) -> list:
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                weight_map = json.load(f).get("weight_map", {})
            files = [
                str(model_path / shard)
                for shard in sorted(set(weight_map.values()))
                if (model_path / shard).exists()
            ]
            if files:
                return files
        except (ValueError, OSError):
            pass
    return [
        path
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def load_encoder_model(
    model_path: Path,
    *,
    model_remapping: Dict[str, str],
    model_class_name: str = "Model",
    config: Optional[dict] = None,
    config_overrides: Optional[dict] = None,
    lazy: bool = False,
    **kwargs,
) -> nn.Module:
    strict = kwargs.pop("strict", True)
    config = dict(config) if config is not None else load_config(model_path, **kwargs)
    if config_overrides:
        config.update(config_overrides)

    model_type = str(config.get("model_type", "")).lower()
    config["model_type"] = model_remapping.get(model_type, model_type)

    weight_files = _weight_files(model_path)
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for weight_file in weight_files:
        weights.update(_load_safetensors(weight_file))
    for weight_file in sorted(glob.glob(str(model_path / "*" / "*.safetensors"))):
        folder = Path(weight_file).parent.name
        for key, value in _load_safetensors(weight_file).items():
            weights[f"{folder}.{key}"] = value

    model_module, _ = get_model_and_args(config=config)
    model_class = getattr(model_module, model_class_name, None)
    if model_class is None:
        raise ValueError(
            f"Model type {config['model_type']!r} does not support {model_class_name}."
        )

    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_module.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config,
        model_module,
        config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    model_config = apply_generation_config_defaults(model_config, config)
    model = model_class(model_config)

    weights = sanitize_weights(model, weights)
    if hasattr(model_module, "VisionModel") and hasattr(model_config, "vision_config"):
        weights = sanitize_weights(
            model_module.VisionModel, weights, model_config.vision_config
        )
    if hasattr(model_module, "LanguageModel") and hasattr(model_config, "text_config"):
        weights = sanitize_weights(
            model_module.LanguageModel, weights, model_config.text_config
        )

    quantization = config.get("quantization")
    if quantization is not None:

        def quantization_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if hasattr(module, "weight") and module.weight.size % 64 != 0:
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=quantization_predicate,
        )

    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    model.model_path = model_path
    model.eval()
    return model
