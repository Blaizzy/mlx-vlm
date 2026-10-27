# Copyright © 2026 Apple Inc.

"""Fused Metal kernels, one kernel per file."""

from .expert_routing import expert_routing
from .moe_softmax_topk import moe_softmax_topk
from .moe_swiglu import moe_swiglu_4bit
from .norm_add_norm import norm_add_norm, pre_norm
from .post_expert_norm_combine import post_expert_norm_combine
from .qk_norm import qk_norm
from .qk_norm_rope import qk_norm_rope
from .rms_norm_sigmoid_gate import rms_norm_sigmoid_gate

__all__ = [
    "expert_routing",
    "moe_softmax_topk",
    "moe_swiglu_4bit",
    "norm_add_norm",
    "post_expert_norm_combine",
    "pre_norm",
    "qk_norm",
    "qk_norm_rope",
    "rms_norm_sigmoid_gate",
]
