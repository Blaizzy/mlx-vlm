"""Reusable task-shaped datasets and collation for implemented trainers."""

from .preference import PreferenceVisionDataset
from .next_token import VisionDataset

__all__ = ["PreferenceVisionDataset", "VisionDataset"]
