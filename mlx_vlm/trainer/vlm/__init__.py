"""Vision-language training recipes over shared datasets and configuration."""

from ..datasets import PreferenceVisionDataset, VisionDataset
from .orpo.trainer import ORPOTrainer, ORPOTrainingArgs, train_orpo
from .sft.trainer import SFTTrainer, TrainingArgs, train
