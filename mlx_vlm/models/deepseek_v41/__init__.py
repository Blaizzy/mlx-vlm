import mlx_vlm.models.deepseek_v41.processing_deepseek_v41  # noqa: F401 (installs processor patch)

from .config import ModelConfig
from .deepseek_v41 import Model
from .dequant import (
    dequant_fp4,
    dequant_fp8,
    dequant_fp8_rows,
    e8m0_to_float,
    is_fp4_expert,
    unpack_fp4,
)
from .dspark import DSparkConfidenceHead, DSparkMarkovHead, get_dspark_topk_idxs
from .engram import Engram, EngramEmbedding, EngramLayout, NgramHashState
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
    SharedIndexState,
    hc_mix_coeffs,
    make_identity_pre_mix,
    select_candidate_blocks,
)
from .vision import Aligner, ViT

__all__ = [
    "Model",
    "ModelConfig",
    "dequant_fp4",
    "dequant_fp8",
    "dequant_fp8_rows",
    "e8m0_to_float",
    "is_fp4_expert",
    "unpack_fp4",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "get_dspark_topk_idxs",
    "Compressor",
    "DeepseekV41Attention",
    "DeepseekV41Block",
    "DeepseekV41Cache",
    "DeepseekV41MoE",
    "DeepseekV41MoEGate",
    "Engram",
    "EngramEmbedding",
    "EngramLayout",
    "NgramHashState",
    "Indexer",
    "LanguageModel",
    "ParallelHead",
    "SharedIndexState",
    "hc_mix_coeffs",
    "make_identity_pre_mix",
    "select_candidate_blocks",
    "Aligner",
    "ViT",
]
