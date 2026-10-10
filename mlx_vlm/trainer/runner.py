"""Shared training and evaluation lifecycle, implemented by the common engine."""

from .common.engine import evaluate, train

__all__ = ["evaluate", "train"]
