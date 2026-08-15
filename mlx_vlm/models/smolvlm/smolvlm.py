from ..idefics3 import Model as Idefics3Model
from . import processing_smolvlm  # noqa: F401


class Model(Idefics3Model):
    """SmolVLM uses Idefics3's multimodal feature layout and model path."""
