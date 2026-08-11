import glob
import json
from pathlib import Path

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

EMBEDDING_MODEL_REMAPPING = {
    "qwen3": "qwen3_embedding",
    "gemma3_text": "gemma3_embedding",
    "lfm2": "lfm2_embedding",
    "ministral3": "ministral3_embedding",
    "xlm-roberta": "xlm_roberta",
}


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
        wf
        for wf in glob.glob(str(model_path / "*.safetensors"))
        if not wf.endswith("consolidated.safetensors")
    ]


def load_embedding_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    strict = kwargs.pop("strict", True)
    config = load_config(model_path, **kwargs)

    model_type = str(config.get("model_type", "")).lower()
    config["model_type"] = EMBEDDING_MODEL_REMAPPING.get(model_type, model_type)

    weight_files = _weight_files(model_path)
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(_load_safetensors(wf))
    for wf in sorted(glob.glob(str(model_path / "*" / "*.safetensors"))):
        folder = Path(wf).parent.name
        for k, v in _load_safetensors(wf).items():
            weights[f"{folder}.{k}"] = v

    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config,
        model_class,
        config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    model_config = apply_generation_config_defaults(model_config, config)

    model = model_class.Model(model_config)

    weights = sanitize_weights(model, weights)
    if hasattr(model_class, "VisionModel") and hasattr(model_config, "vision_config"):
        weights = sanitize_weights(
            model_class.VisionModel, weights, model_config.vision_config
        )
    if hasattr(model_class, "LanguageModel") and hasattr(model_config, "text_config"):
        weights = sanitize_weights(
            model_class.LanguageModel, weights, model_config.text_config
        )

    quantization = config.get("quantization", None)
    if quantization is not None:

        def _quant_predicate(path, module):
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
            class_predicate=_quant_predicate,
        )

    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())
    model.model_path = model_path
    model.eval()
    return model
