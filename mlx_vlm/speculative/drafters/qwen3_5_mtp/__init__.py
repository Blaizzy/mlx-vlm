from .compact_head import CompactProposalHead, load_compact_proposal_head
from .config import Qwen3_5MTPConfig as ModelConfig
from .config import TextConfig
from .qwen3_5_mtp import Qwen3_5MTPDraftModel
from .qwen3_5_mtp import Qwen3_5MTPDraftModel as Model

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "CompactProposalHead",
    "load_compact_proposal_head",
    "Qwen3_5MTPDraftModel",
]
