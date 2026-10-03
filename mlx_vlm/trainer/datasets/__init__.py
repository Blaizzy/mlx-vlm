"""Reusable task-shaped datasets and collation for implemented trainers."""

from .next_token import VisionDataset
from .preference import PreferenceVisionDataset

__all__ = ["PreferenceVisionDataset", "VisionDataset"]
