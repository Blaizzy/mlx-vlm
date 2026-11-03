import mlx.core as mx

from ..minicpmv4_6.language import LanguageModel as MiniCPMVLanguageModel
from ..qwen3_5_moe.language import LanguageModel as Qwen35MoELanguageModel


class LanguageModel(MiniCPMVLanguageModel):
    def get_rope_index(self, input_ids, *args, **kwargs):
        # APC cannot rebuild a canvas from token IDs without the crop grids.
        visual = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
        )
        if mx.any(visual).item():
            raise ValueError(
                "MiniCPM-V 4.7 visual positions require crop grids from get_input_embeddings."
            )
        return super().get_rope_index(input_ids, *args, **kwargs)


class MoELanguageModel(Qwen35MoELanguageModel, LanguageModel):
    """Qwen3.5 experts with the MiniCPM canvas-position contract."""
