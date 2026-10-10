"""Vision-language training recipes over shared datasets and configuration."""

from ..datasets import PreferenceVisionDataset, VisionDataset
from .config import VLMTrainingArgs
from .dpo.config import DPOTrainingArgs
from .orpo.config import ORPOTrainingArgs
from .orpo.trainer import ORPOTrainer, train_orpo
from .orpo.trainer import ORPOTrainingArgs as LegacyORPOTrainingArgs
from .sft.trainer import SFTTrainer, TrainingArgs, train
