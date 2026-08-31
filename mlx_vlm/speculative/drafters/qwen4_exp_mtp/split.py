from types import SimpleNamespace
from typing import Dict

import mlx.core as mx

from ....models.qwen4_exp.config import TextConfig
from ....models.qwen4_exp.fp8 import convert_qwen4_exp_fp8_weights
from ..mtp_split import MTPSplitter
from .qwen4_exp_mtp import Qwen4ExpMTPDraftModel


class Qwen4ExpMTPSplitter(MTPSplitter):
    output_model_type = "qwen4_exp_mtp"
    draft_model_cls = Qwen4ExpMTPDraftModel
    tie_word_embeddings_default = False
    depth_field = "mtp_num_hidden_layers"
    block_size_extra = 1

    def select_keys(self, key: str, text_config: dict) -> bool:
        del text_config
        return key.startswith(("mtp.", "language_model.mtp.", "model.mtp."))

    def should_quantize_key(self, key: str) -> bool:
        return super().should_quantize_key(key) and not key.endswith(
            "layers.0.mlp.gate.weight"
        )

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        del text_config
        return convert_qwen4_exp_fp8_weights(tensors)

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(
            config=SimpleNamespace(text_config=TextConfig.from_dict(text_config))
        )


def split_qwen4_exp_mtp(source: str, output: str, **kwargs):
    return Qwen4ExpMTPSplitter().split(source, output, **kwargs)
