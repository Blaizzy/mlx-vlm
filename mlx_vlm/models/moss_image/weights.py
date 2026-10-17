from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from ..flux2.vae import Flux2VAE
from ..qwen3_5.config import TextConfig
from ..qwen3_5.language import Qwen3_5Model
from .config import MOSSImageTransformerConfig
from .transformer import MOSSImageTransformer


def read_config(path):
    return json.loads(Path(path).read_text())


def load_safetensors(directory):
    directory = Path(directory)
    files = sorted(
        file
        for file in directory.glob("*.safetensors")
        if not file.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    weights = {}
    for file in files:
        weights.update(mx.load(str(file)))
    return weights


def sanitize_text_encoder_weights(weights):
    result = {}
    offset_norms = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for key, value in weights.items():
        # The upstream AutoModel includes a vision tower; text-to-image uses
        # only its language backbone and its final normalized hidden states.
        if not key.startswith("language_model."):
            continue
        key = key.removeprefix("language_model.")
        if key.endswith("conv1d.weight"):
            value = value.moveaxis(2, 1)
        if key == "norm.weight" or key.endswith(offset_norms):
            value = value + 1.0
        result[key] = value
    return result


def load_text_encoder(path):
    path = Path(path) / "text_encoder"
    metadata = read_config(path / "config.json")
    config = TextConfig.from_dict(metadata["text_config"])
    model = Qwen3_5Model(config)
    weights = load_safetensors(path)
    if metadata.get("mlx_vlm_format") != "moss_image":
        weights = sanitize_text_encoder_weights(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def load_transformer(path):
    path = Path(path) / "transformer"
    config = MOSSImageTransformerConfig.from_dict(read_config(path / "config.json"))
    model = MOSSImageTransformer(config)
    weights = model.sanitize(load_safetensors(path))
    model.load_weights(list(weights.items()), strict=True)
    return model


def load_vae(path):
    path = Path(path) / "vae"
    config = read_config(path / "config.json")
    model = Flux2VAE(
        decoder_block_out_channels=tuple(config["decoder_block_out_channels"])
    )
    model.bn.eps = config.get("batch_norm_eps", 1e-4)
    weights = {}
    for key, value in load_safetensors(path).items():
        if not key.startswith(("decoder.", "post_quant_conv.", "bn.")) or key.endswith(
            "num_batches_tracked"
        ):
            continue
        key = key.replace(".to_out.0.", ".to_out.")
        if value.ndim == 4 and config.get("mlx_vlm_format") != "moss_image":
            value = value.transpose(0, 2, 3, 1)
        weights[key] = value.astype(mx.float32)
    model.load_weights(list(weights.items()), strict=True)
    return model
