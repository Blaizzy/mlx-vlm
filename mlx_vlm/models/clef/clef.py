import mlx.core as mx

from ..base import InputEmbeddingsFeatures
from ..qwen3_5.qwen3_5 import Model as QwenModel
from .head import JointSchemaHead


class Model(QwenModel):
    def __init__(self, config):
        super().__init__(config)
        self.head = JointSchemaHead(**config.joint_head_config)

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Clef is a decision model; use /v1/systemone or mlx_vlm.decision"
        )

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        """Scatter each modality separately, while sharing multimodal positions."""
        embeds = self.language_model.model.embed_tokens(input_ids)
        for pixels, grid, token in (
            (pixel_values, kwargs.get("image_grid_thw"), self.config.image_token_index),
            (
                kwargs.get("pixel_values_videos"),
                kwargs.get("video_grid_thw"),
                self.config.video_token_index,
            ),
        ):
            if pixels is not None:
                dtype = self.vision_tower.patch_embed.proj.weight.dtype
                features, _ = self.vision_tower(pixels.astype(dtype), grid)
                embeds, _ = self.merge_input_ids_with_image_features(
                    features, embeds, input_ids, token, token
                )
        positions, deltas = self.language_model.get_rope_index(
            input_ids,
            kwargs.get("image_grid_thw"),
            kwargs.get("video_grid_thw"),
            kwargs.get("mask"),
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=embeds, position_ids=positions, rope_deltas=deltas
        )

    def decide(self, record):
        input_ids = mx.array(record.input_ids)[None]
        media = {
            key: mx.array(value)
            for key, value in (record.media or {}).items()
            if key
            in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            )
        }
        features = self.get_input_embeddings(input_ids, **media)
        hidden = self.language_model.model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
        )
        return self.head(hidden, input_ids, record, self._output_embeddings)

    def _output_embeddings(self, ids):
        layer = (
            self.language_model.model.embed_tokens
            if self.config.text_config.tie_word_embeddings
            else self.language_model.lm_head
        )
        if hasattr(layer, "scales"):
            # Gather first: never dequantize the full vocabulary matrix.
            biases = getattr(layer, "biases", None)
            return mx.dequantize(
                layer.weight[ids],
                layer.scales[ids],
                biases[ids] if biases is not None else None,
                group_size=layer.group_size,
                bits=layer.bits,
                mode=layer.mode,
            )
        return layer.weight[ids]

    def sanitize(self, weights):
        head = {key: value for key, value in weights.items() if key.startswith("head.")}
        backbone = super().sanitize(
            {
                key: value
                for key, value in weights.items()
                if not key.startswith("head.")
            }
        )
        # Run the head in the backbone dtype, as the release loader does.
        dtype = backbone["language_model.model.embed_tokens.weight"].dtype
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            dtype = head["head.hidden_norm.weight"].dtype
        # MLX Sequential stores its modules under `layers`.
        for key, value in head.items():
            if ".feedforward." in key and ".feedforward.layers." not in key:
                key = key.replace(".feedforward.", ".feedforward.layers.")
            if key.startswith("head.residual_scorer.") and not key.startswith(
                "head.residual_scorer.layers."
            ):
                key = key.replace(
                    "head.residual_scorer.", "head.residual_scorer.layers."
                )
            backbone[key] = value.astype(dtype)
        return backbone

    @property
    def quant_predicate(self):
        base = super().quant_predicate
        lexical_layer = (
            "language_model.model.embed_tokens"
            if self.config.text_config.tie_word_embeddings
            else "language_model.lm_head"
        )

        def predicate(path, module):
            if path.startswith("head."):
                return False
            if (
                self.config.decision_quantization == "preserve-output"
                and path == lexical_layer
            ):
                return False
            return base(path, module) if base else True

        return predicate
