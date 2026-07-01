import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from ....models.base import BaseModelConfig


@dataclass
class DeepseekV4DSparkConfig(BaseModelConfig):
    """Config for the DSpark self-speculative drafter over a DeepSeek-V4 target.

    The DeepSeek-V4-{Flash,Pro}-DSpark checkpoints ship the draft stack under the
    ``mtp.*`` namespace of the bundled fp8/fp4 weights and declare the DSpark draft
    hyper-parameters (``dspark_block_size`` / ``dspark_target_layer_ids`` / ...) on the
    base ``deepseek_v4`` config. Field names follow the HF/mlx-vlm target config
    (``hidden_size`` / ``num_attention_heads`` style) so the reused primitives —
    :class:`DeepseekV4MoE`, :class:`HyperConnection`, :class:`HyperHead` — read it
    directly. The same config drives both Flash and Pro (only the dims differ).
    """

    model_type: str = "deepseek_v4_dspark"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    attention_bias: bool = False
    sliding_window: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1048576

    # MoE (consumed by the reused DeepseekV4MoE / MoEGate)
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0

    # Hyper-Connections (consumed by the reused HyperConnection / HyperHead)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # DSpark draft stack
    num_hidden_layers: int = (
        43  # base depth; draft layer_id = num_hidden_layers + stage
    )
    n_mtp_layers: int = 3
    block_size: int = 5
    noise_token_id: int = 128799
    markov_rank: int = 256
    target_layer_ids: List[int] = field(default_factory=lambda: [40, 41, 42])
    tie_word_embeddings: bool = False
    runtime_block_size: Optional[int] = None

    @property
    def fc_in(self) -> int:
        """Input width of the stage-0 ``main_proj`` (concatenated target hiddens)."""
        return self.hidden_size * len(self.target_layer_ids)

    @property
    def confidence_in(self) -> int:
        return self.hidden_size + self.markov_rank

    @classmethod
    def from_dict(cls, params: dict) -> "DeepseekV4DSparkConfig":
        flat = dict(params)
        # DSpark draft hyper-parameters are namespaced ``dspark_*`` on the base config.
        alias = {
            "dspark_block_size": "block_size",
            "dspark_noise_token_id": "noise_token_id",
            "dspark_markov_rank": "markov_rank",
            "dspark_target_layer_ids": "target_layer_ids",
            "sliding_window": "sliding_window",
        }
        for src, dst in alias.items():
            if src in flat and dst not in flat:
                flat[dst] = flat[src]
        # The published configs only carry the base-MTP depth (num_nextn_predict_layers);
        # the DSpark draft depth is a distinct field (default 3) confirmable from the
        # checkpoint's mtp.N count at load.
        if "n_mtp_layers" not in flat and flat.get("num_nextn_predict_layers"):
            flat["n_mtp_layers"] = flat["num_nextn_predict_layers"]
        sig = inspect.signature(cls).parameters
        kwargs = {k: v for k, v in flat.items() if k in sig}
        if "target_layer_ids" in kwargs:
            kwargs["target_layer_ids"] = list(kwargs["target_layer_ids"])
        return cls(**kwargs)

    from_hf_dict = from_dict
