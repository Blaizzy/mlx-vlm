# Import LanguageModel and VisionModel from idefics3 since smolvlm uses them directly
import mlx_vlm.models.smolvlm.processing_smolvlm  # noqa: F401 (installs processor patch)

from ..idefics3 import LanguageModel, VisionModel
from .config import ModelConfig, TextConfig, VisionConfig
from .smolvlm import Model
