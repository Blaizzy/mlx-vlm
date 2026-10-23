"""Hi3D v1.0: image-conditioned high-resolution mesh refinement (Sparc3D)."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .decoder import VAE
from .transformer import Hi3DTransformer
from .vision import Conditioner


def read_weights(root):
    """Load every safetensors shard a model directory's index lists."""
    root = Path(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    weights = {}
    for name in files:
        if Path(name).name != name:
            raise ValueError("Checkpoint shards must be in the model directory")
        shard = mx.load(str(root / name))
        if weights.keys() & shard.keys():
            raise ValueError("Duplicate tensor names across checkpoint shards")
        weights.update(shard)
    return weights


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer_512 = Hi3DTransformer(config.transformer_512)
        self.transformer_1024 = Hi3DTransformer(config.transformer_1024)
        self.vae = VAE(config.decoder)
        self.conditioner = Conditioner(config)
        self.eval()
        self.freeze()

    def train(self, mode=True):
        if mode:
            raise ValueError("Hi3D is an inference-only model")
        return super().train(False)

    @classmethod
    def from_pretrained(cls, path):
        """Load a local converted bundle with strict weights."""
        root = Path(path)
        model = cls(ModelConfig.load(root / "config.json"))
        weights = model.sanitize(read_weights(root))
        model.load_weights(list(weights.items()), strict=True)
        model.freeze()
        model.eval()
        mx.eval(model.parameters())
        return model

    @staticmethod
    def sanitize(weights):
        # The octree child offsets are constants of the decoder, not weights.
        return {k: v for k, v in weights.items() if not k.endswith("inv_grid_coord")}

    def __call__(self, image, vertices, faces, **kwargs):
        from .pipeline import Pipeline

        return Pipeline(self).generate(image, vertices, faces, **kwargs)
