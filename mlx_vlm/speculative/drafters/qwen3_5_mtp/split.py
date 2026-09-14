from typing import Dict

import mlx.core as mx

from ....fp8 import make_quantization_config
from ..mtp_split import MTPSplitter
from .qwen3_5_mtp import Qwen3_5MTPDraftModel


class Qwen3_5MTPSplitter(MTPSplitter):
    output_model_type = "qwen3_5_mtp"
    draft_model_cls = Qwen3_5MTPDraftModel
    tie_word_embeddings_default = True
    depth_field = "mtp_num_hidden_layers"
    block_size_extra = 1
    supports_mlx_source = True

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith("mtp.")

    def on_mlx_source(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return {
            (key[len("mtp.") :] if key.startswith("mtp.") else key): value
            for key, value in tensors.items()
        }

    def quantization_from_source(self, tensors, source_config):
        if not any(key.endswith(".scales") for key in tensors):
            return None
        quantization = source_config.get("mtplx_mtp_quantization")
        if quantization is None:
            quantization = source_config.get("quantization")
        if quantization is None:
            quantization = make_quantization_config(source_config)
        return quantization
