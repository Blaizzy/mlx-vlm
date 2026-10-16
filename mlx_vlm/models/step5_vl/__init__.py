from ..base import install_auto_processor_patch
from ..step3p7.processing_step3p7 import Step3VLProcessor as Step5VLProcessor
from ..step3p7.vision import VisionModel
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .step5_vl import Model

install_auto_processor_patch("step5_vl", Step5VLProcessor)
