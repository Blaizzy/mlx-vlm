"""Vision-language training recipes over shared datasets and configuration."""

from ..datasets import PreferenceVisionDataset, VisionDataset
from .config import SFTTrainConfig, VLMTrainConfig, VLMTrainingArgs
from .dpo.config import DPOTrainConfig, DPOTrainingArgs
from .orpo.config import ORPOTrainConfig, ORPOTrainingArgs
from .orpo.trainer import ORPOTrainer
from .orpo.trainer import ORPOTrainingArgs as LegacyORPOTrainingArgs
from .orpo.trainer import train_orpo
from .sft.trainer import SFTTrainer, TrainingArgs, train
