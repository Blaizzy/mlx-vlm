"""Frozen inference models for SAM 3D Objects."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig
from .decoders import GaussianDecoder, MeshDecoder, StructureDecoder
from .flow import LatentFlow, StructureFlow
from .vision import ConditionEncoder


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ss_generator = StructureFlow(config)
        self.slat_generator = LatentFlow(config)
        self.ss_condition_embedder = ConditionEncoder(config, pointmap=True)
        self.slat_condition_embedder = ConditionEncoder(config)
        self.ss_decoder = StructureDecoder(config)
        self.slat_decoder_gs = GaussianDecoder(config)
        self.slat_decoder_gs_4 = GaussianDecoder(config, count=4)
        self.slat_decoder_mesh = MeshDecoder(config)
        self.shared_backbone = False
        if config.depth_model is not None:
            from ..moge3.moge3 import Model as DepthModel

            self.depth_model = DepthModel(config.depth_model)
        self.eval()
        self.freeze()

    def train(self, mode=True):
        if mode:
            raise ValueError("SAM 3D Objects is an inference-only model")
        return super().train(False)

    @classmethod
    def from_pretrained(cls, path):
        """Load a local converted bundle using MLX alone, with strict weights."""
        root = Path(path)
        config = ModelConfig.load(root / "config.json")
        model = cls(config)
        weights = read_weights(root)
        model.load_weights(list(weights.items()), strict=True)
        model.freeze()
        model.eval()
        del weights
        model.share_backbones()
        mx.async_eval(model.parameters())
        return model

    def share_backbones(self):
        if self.shared_backbone:
            return True
        wrappers = (
            self.ss_condition_embedder.module_list[:2]
            + self.slat_condition_embedder.module_list[:2]
        )
        reference = tree_flatten(wrappers[0].backbone.parameters())
        checks = []
        for wrapper in wrappers[1:]:
            params = tree_flatten(wrapper.backbone.parameters())
            if [k for k, _ in params] != [k for k, _ in reference] or any(
                a.shape != b.shape or a.dtype != b.dtype
                for (_, a), (_, b) in zip(params, reference)
            ):
                return False
            checks.extend(
                mx.array_equal(a, b) for (_, a), (_, b) in zip(params, reference)
            )
        if not bool(mx.all(mx.stack(checks)).item()):
            return False
        for wrapper in wrappers[1:]:
            wrapper.backbone = wrappers[0].backbone
        self.shared_backbone = True
        return True

    @staticmethod
    def sanitize(weights):
        return weights

    def __call__(self, *args, **kwargs):
        from .pipeline import Pipeline

        return Pipeline(self).generate(*args, **kwargs)
