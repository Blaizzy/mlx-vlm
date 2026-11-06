from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn

from ....models.base import BaseModelConfig, create_attention_mask
from ....models.cache import KVCache
from ....models.mellum.config import ModelConfig as TextConfig
from ....models.mellum.language import MellumDecoderLayer
from ..mtp_base import AutoregressiveMTPDraftModel


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "mellum_mtp"
    text_config: TextConfig | None = None
    block_size: int = 2

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


class MellumMTPDraftModel(AutoregressiveMTPDraftModel):
    """Mellum's native next-token block, fed by post-norm target hidden states."""

    def __init__(self, config):
        super().__init__(config)
        text = config.text_config
        if text.num_nextn_predict_layers != 1:
            raise ValueError("Mellum MTP requires exactly one prediction layer.")
        # The appended block continues the target's repeating SWA pattern.
        self.layer_type = text.layer_types[0]
        layer_config = replace(text, num_hidden_layers=1, layer_types=[self.layer_type])
        self.fc = nn.Linear(2 * text.hidden_size, text.hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.layers = [MellumDecoderLayer(layer_config, 0)]
        self.norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)

    def make_cache(self):
        # A trim-safe cache keeps rejected draft tokens from evicting accepted KV.
        return [KVCache()]

    def _forward_hidden(self, token_embed, hidden, tokens, cache, **kwargs):
        h = self.fc(
            mx.concatenate(
                [
                    self.pre_fc_norm_embedding(token_embed),
                    self.pre_fc_norm_hidden(hidden),
                ],
                axis=-1,
            )
        )
        window = (
            self.config.text_config.sliding_window
            if self.layer_type == "sliding_attention"
            else None
        )
        mask = create_attention_mask(h, cache[0], window_size=window)
        h = self.norm(self.layers[0](h, mask, cache[0]))
        return h, h

    def sanitize(self, weights):
        from ..qwen3_5_mtp.split import Qwen3NextMTPSplitter

        weights = {k.removeprefix("mtp."): v for k, v in weights.items()}
        Qwen3NextMTPSplitter().postprocess(weights, self.config.text_config.to_dict())
        return weights
