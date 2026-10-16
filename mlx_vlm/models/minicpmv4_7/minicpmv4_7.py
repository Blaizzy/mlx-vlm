import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..minicpmv4_6.minicpmv4_6 import Merger
from ..minicpmv4_6.minicpmv4_6 import Model as MiniCPMVModel
from ..minicpmv4_6.minicpmv4_6 import VitMerger
from ..minicpmv4_6.vision import VisionModel
from ..qwen3_5_moe.qwen3_5_moe import Model as Qwen35MoEModel
from .config import ModelConfig
from .language import LanguageModel, MoELanguageModel
from .rope import CanvasRoPE


class Model(MiniCPMVModel):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        language_cls = (
            MoELanguageModel
            if config.text_config.model_type == "qwen3_5_moe_text"
            else LanguageModel
        )
        self.language_model = language_cls(config.text_config, config)
        self.vit_merger = VitMerger(
            config.vision_config.hidden_size,
            config.vision_config.window_intermediate_size,
            config.vision_config.num_attention_heads,
            config.vision_config.window_kernel_size,
        )
        self.merger = Merger(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.merger_times,
            config.merge_kernel_size,
        )
        self.vit_merger.act = nn.GELU(approx="tanh")

    def sanitize(self, weights):
        if self.config.text_config.model_type == "qwen3_5_moe_text":
            # Split fused experts; copy array handles before in-place norm updates.
            weights = Qwen35MoEModel.sanitize(
                self, {key: mx.array(value) for key, value in weights.items()}
            )
        return super().sanitize(weights)

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        mode = kwargs.get("downsample_mode", self.config.downsample_mode)
        if mode not in ("4x", "16x"):
            raise ValueError("downsample_mode must be '4x' or '16x'.")
        ids = np.asarray(input_ids)
        modalities = np.where(
            ids == self.config.image_token_id,
            1,
            np.where(ids == self.config.video_token_id, 2, 0),
        )
        sizes = {1: [], 2: []}
        targets = kwargs.get("tgt_sizes")
        bounds = []
        factor = 2 if mode == "4x" else 4
        for batch_idx, row in enumerate(modalities):
            starts = np.flatnonzero(
                (row != 0) & (np.concatenate(([0], row[:-1])) != row)
            )
            ends = (
                np.flatnonzero((row != 0) & (np.concatenate((row[1:], [0])) != row)) + 1
            )
            bounds.append(np.stack([starts, ends], axis=-1))
            grids = (
                np.asarray(targets[batch_idx]).reshape(-1, 2)
                if targets is not None
                else np.empty((0, 2), dtype=np.int32)
            )
            if len(grids) != len(starts):
                raise ValueError(
                    "MiniCPM-V 4.7 crop grids must match visual placeholder spans."
                )
            for start, end, grid in zip(starts, ends, grids):
                if (
                    np.any(grid <= 0)
                    or np.any(grid % factor)
                    or int(np.prod(grid // factor)) != end - start
                ):
                    raise ValueError(
                        "MiniCPM-V 4.7 crop grid does not match its visual token count."
                    )
                sizes[int(row[start])].append(grid)

        positions, deltas = CanvasRoPE(self.config).get_rope_index(
            ids,
            mm_token_type_ids=modalities,
            target_sizes=np.asarray(sizes[1]),
            target_sizes_videos=np.asarray(sizes[2]),
            downsample_mode=mode,
            attention_mask=kwargs.get("attention_mask", kwargs.get("mask")),
        )
        if (
            np.any(modalities)
            and pixel_values is None
            and kwargs.get("cached_image_features") is None
        ):
            raise ValueError("Visual placeholders require image or video features.")
        kwargs["image_bound"] = bounds
        embeddings = super().get_input_embeddings(
            input_ids, pixel_values=pixel_values, **kwargs
        )
        # The language cache counts padded tokens, so subtract the padded width.
        deltas = positions.max(axis=(0, 2))[:, None] + 1 - ids.shape[1]
        positions, deltas = mx.array(positions), mx.array(deltas, dtype=mx.int32)
        self.language_model._position_ids = positions
        self.language_model._rope_deltas = deltas
        return InputEmbeddingsFeatures(
            inputs_embeds=embeddings.inputs_embeds,
            position_ids=positions,
            rope_deltas=deltas,
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        attention_cache = cache[self.language_model.model.fa_idx] if cache else None
        offset = getattr(attention_cache, "offset", 0)
        if attention_cache is not None and mx.any(mx.array(offset) > 0).item():
            return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        return self.language_model(
            input_ids, mask=mask, cache=cache, **features.to_dict()
        )
