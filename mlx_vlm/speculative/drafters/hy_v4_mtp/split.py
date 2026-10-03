from types import SimpleNamespace
from typing import Dict

import mlx.core as mx

from ....models.hy_v4.config import ModelConfig as HyV4Config
from ..mtp_split import MTPSplitter
from .hy_v4_mtp import HyV4MTPDraftModel


class HyV4MTPSplitter(MTPSplitter):
    output_model_type = "hy_v4_mtp"
    draft_model_cls = HyV4MTPDraftModel
    require_text_config = False
    tie_word_embeddings_default = False
    depth_field = "num_nextn_predict_layers"
    block_size_extra = 1

    def select_keys(self, key: str, text_config: dict) -> bool:
        del text_config
        return key.startswith("model.mtp_layers.")

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        del text_config
        prefix = "model.mtp_layers.0."
        renamed = {}
        for key, value in tensors.items():
            if not key.startswith(prefix):
                continue
            key = key[len(prefix) :]
            if key.startswith(
                ("input_layernorm.", "post_attention_layernorm.", "self_attn.", "mlp.")
            ):
                key = f"decoder.{key}"
            elif key.startswith("final_layernorm."):
                key = f"norm.{key[len('final_layernorm.'):]}"
            renamed[key] = value
        return renamed

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(args=HyV4Config.from_dict(text_config))

    def quantization_from_source(self, tensors, source_config):
        del source_config
        if any(key.endswith(".scales") for key in tensors):
            return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        return None


def split_hy_v4_mtp(source: str, output: str, **kwargs):
    return HyV4MTPSplitter().split(source, output, **kwargs)
