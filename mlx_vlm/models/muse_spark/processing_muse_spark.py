from ..base import install_auto_processor_patch
from ..muse_glimmer.processing_muse_glimmer import (
    MuseGlimmerImageProcessor,
    MuseGlimmerProcessor,
)
from .config import ModelConfig


class MuseSparkImageProcessor(MuseGlimmerImageProcessor):
    pass


class MuseSparkProcessor(MuseGlimmerProcessor):
    """Muse Glimmer's image processing and chat channels; audio is not supported."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import PreTrainedConfig

        model_config = kwargs.pop("config", None)
        if model_config is None:
            config_dict, _ = PreTrainedConfig.get_config_dict(
                pretrained_model_name_or_path,
                **{k: v for k, v in kwargs.items() if k not in ("use_fast",)},
            )
            model_config = ModelConfig.from_dict(config_dict)
        elif isinstance(model_config, dict):
            model_config = ModelConfig.from_dict(model_config)
        return super().from_pretrained(
            pretrained_model_name_or_path, config=model_config, **kwargs
        )


install_auto_processor_patch("muse_spark", MuseSparkProcessor)
