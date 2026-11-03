from types import SimpleNamespace

from ....utils import _load_safetensors
from ..mtp_split import MTPSplitter
from .config import NemotronHMTPConfig
from .nemotron_h_mtp import NemotronHMTPDraftModel


class NemotronHMTPSplitter(MTPSplitter):
    output_model_type = "nemotron_h_mtp"
    draft_model_cls = NemotronHMTPDraftModel

    def read_text_config(self, source_config):
        return dict(
            source_config.get("text_config")
            or source_config.get("llm_config")
            or source_config
        )

    def select_keys(self, key, text_config):
        return key.startswith(("mtp.", "language_model.mtp."))

    def load_shard(self, file, keys):
        # Also handles safetensors F8_E4M3, which MLX reads as raw bytes.
        shard = _load_safetensors(str(file))
        return {key: shard[key] for key in keys}

    def rename(self, tensors, text_config):
        return {
            key.removeprefix("language_model.").removeprefix("mtp."): value
            for key, value in tensors.items()
        }

    def sanitize_ctx(self, text_config):
        return SimpleNamespace(config=NemotronHMTPConfig(text_config=text_config))

    def quantization_from_source(self, tensors, source_config):
        if any(key.endswith(".scales") for key in tensors):
            return source_config.get("quantization")
        return None

    def should_quantize_key(self, key):
        return key.endswith(".weight") and not key.endswith(".mixer.gate.weight")
