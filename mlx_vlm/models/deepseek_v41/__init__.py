"""DeepSeek-V4.1 model package. Importing it installs the processor patch."""

import mlx_vlm.models.deepseek_v41.processing_deepseek_v41  # noqa: F401

from .config import ModelConfig
from .deepseek_v41 import Model
from .dequant import dequant_fp4, dequant_fp8, e8m0_to_float, is_fp4_expert, unpack_fp4
from .dspark import DSparkConfidenceHead
from .engram import Engram, EngramLayout, NgramHashState
from .fakequant import fake_quant_fp4_e4m3, fake_quant_fp4_ue8m0, fake_quant_fp8_ue8m0
from .language import (
    Compressor,
    DeepseekV41Attention,
    DeepseekV41Block,
    DeepseekV41Cache,
    DeepseekV41MoE,
    DeepseekV41MoEGate,
    Indexer,
    LanguageModel,
    ParallelHead,
    hc_mix_coeffs,
    make_identity_pre_mix,
    sanitize_moe_weights,
    select_candidate_blocks,
)
from .vision import Aligner, ViT

__all__ = [
    "Model",
    "ModelConfig",
    "dequant_fp4",
    "dequant_fp8",
    "e8m0_to_float",
    "is_fp4_expert",
    "unpack_fp4",
    "DSparkConfidenceHead",
    "Compressor",
    "DeepseekV41Attention",
    "DeepseekV41Block",
    "DeepseekV41Cache",
    "DeepseekV41MoE",
    "DeepseekV41MoEGate",
    "Engram",
    "EngramLayout",
    "NgramHashState",
    "fake_quant_fp4_e4m3",
    "fake_quant_fp4_ue8m0",
    "fake_quant_fp8_ue8m0",
    "Indexer",
    "LanguageModel",
    "ParallelHead",
    "hc_mix_coeffs",
    "make_identity_pre_mix",
    "sanitize_moe_weights",
    "select_candidate_blocks",
    "Aligner",
    "ViT",
]
