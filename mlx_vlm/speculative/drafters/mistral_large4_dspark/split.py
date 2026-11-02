import json

from ..mtp_split import MTPSplitter
from .dspark import Model


class MistralLarge4DsparkSplitter(MTPSplitter):
    output_model_type = "mistral_large4_dspark"
    draft_model_cls = Model

    def select_keys(self, key, text_config):
        return key.startswith("mtp_dspark.")

    def depth(self, text_config):
        return int(text_config["mtp_config"]["depth"])

    def prepare(self, source_path, *, source_config=None, **kwargs):
        if source_config is None:
            source_config = json.loads((source_path / "config.json").read_text())
        source_config = dict(source_config)
        text = self.read_text_config(source_config)
        # Native Mistral releases keep the draft metadata in params.json.
        if not text.get("mtp_config"):
            params_path = source_path / "params.json"
            if params_path.exists():
                text["mtp_config"] = json.loads(params_path.read_text()).get("mtp")
        if not (text.get("mtp_config") or {}).get("dspark"):
            raise ValueError(
                "No Mistral Large 4 DSpark metadata in config.json or params.json"
            )
        source_config["text_config"] = text
        return super().prepare(source_path, source_config=source_config, **kwargs)
