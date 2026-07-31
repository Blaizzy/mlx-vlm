from .config import ModelConfig
from .inkling import Model
from .language import LanguageModel

__all__ = ["Model", "ModelConfig", "LanguageModel"]


def _patch_hf_processor_dmel_numpy():
    """Let InklingProcessor._extract_dmel_bins accept numpy input.

    The HF processor assumes torch tensors, but this pipeline runs the
    feature extractor with numpy return types, which crashes the dMel bin
    extraction with `'numpy.ndarray' object has no attribute 'to'`. Coerce
    numpy -> torch for the call and hand back numpy when numpy came in, so
    the surrounding BatchFeature conversion stays type-consistent.
    """
    try:
        import numpy as np
        from transformers.models.inkling.processing_inkling import InklingProcessor
    except Exception:  # transformers without inkling — nothing to patch
        return
    orig = InklingProcessor._extract_dmel_bins
    if getattr(orig, "_mlx_vlm_numpy_ok", False):
        return

    def _extract_dmel_bins(self, input_features):
        import torch

        was_numpy = isinstance(input_features, np.ndarray)
        if was_numpy:
            input_features = torch.from_numpy(input_features)
        out = orig(self, input_features)
        return out.numpy() if was_numpy else out

    _extract_dmel_bins._mlx_vlm_numpy_ok = True
    InklingProcessor._extract_dmel_bins = _extract_dmel_bins


_patch_hf_processor_dmel_numpy()
