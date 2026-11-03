from pathlib import Path

import mlx.core as mx

from .text_encoder import Flux3TextEncoder, text_config
from .transformer import Flux3Transformer
from .vae import Flux3VAE


def load_transformer(path, config):
    model = Flux3Transformer(config)
    weights = mx.load(str(Path(path) / "model.safetensors"))
    weights = {
        k.replace("adaLN_modulation.1.", "adaLN_modulation.layers.1."): v.astype(
            mx.bfloat16
        )
        for k, v in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    return model


def load_text_encoder(path, layers):
    path = Path(path) / "text_encoder"
    config = text_config(path)
    if not layers or min(layers) < 1 or max(layers) > config.num_hidden_layers:
        raise ValueError("Flux3 text feature layers are outside the text backbone")
    model = Flux3TextEncoder(config)
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        for key, value in mx.load(str(file)).items():
            prefix = "model.language_model."
            if key.startswith(prefix):
                weights[key.removeprefix(prefix)] = value.astype(mx.bfloat16)
    model.load_weights(list(weights.items()), strict=True)
    # Layers after the last requested feature and final norm are unused. Keep
    # the norm only when extracting the final normalized hidden state.
    if max(layers) < config.num_hidden_layers:
        model.layers = model.layers[: max(layers)]
        model.norm = None
    return model


def load_vae(path):
    model = Flux3VAE()
    weights = mx.load(str(Path(path) / "ae.safetensors"))
    weights = {
        k: (v.transpose(0, 2, 3, 1) if v.ndim == 4 else v).astype(mx.float32)
        for k, v in weights.items()
        if not k.endswith("num_batches_tracked")
    }
    model.load_weights(list(weights.items()), strict=True)
    return model
