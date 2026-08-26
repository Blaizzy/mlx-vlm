from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_5.config import resolve_qwen_eos_token_id, sanitize_quantization_config
from ..qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config

LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"


@dataclass
class VisionConfig(Qwen3VLVisionConfig):
    model_type: str = "qwen4_exp"

    def __post_init__(self):
        if self.deepstack_visual_indexes:
            raise ValueError(
                "deepstack is not used by qwen4_exp, but deepstack_visual_indexes "
                f"is set to {self.deepstack_visual_indexes}"
            )
        self.deepstack_visual_indexes = []


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    norm_topk_prob: bool = True
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        }
    )
    # Layer schedule: linear attention layers interleaved with sparse-attention ones.
    layer_types: Optional[List[str]] = None
    deepseek_sparse_attention_interval: int = 4
    # Gated delta net output gate ("sigmoid" or "silu"); falls back to hidden_act.
    output_gate_type: Optional[str] = None
    # Hyper-connections.
    hc_count: int = 4
    hc_lowrank: int = 320
    # Per-Layer Embedding (PLE).
    ple_layer_ids: Optional[List[int]] = None
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 512
    # QSA indexer (sparse attention layers).
    indexer_n_heads: Optional[int] = None
    indexer_kv_heads: Optional[int] = None
    indexer_head_dim: Optional[int] = None
    indexer_budget: Optional[int] = None
    indexer_compress_ratio: Optional[int] = None

    def __post_init__(self):
        if self.rope_parameters:
            # Normalize rope_parameters keys (accept both 'rope_type' and 'type')
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            required_keys = {
                "mrope_section",
                "type",
                "rope_theta",
                "partial_rotary_factor",
            }
            if not all(key in self.rope_parameters for key in required_keys):
                raise ValueError(f"rope_parameters must contain keys {required_keys}")

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.layer_types is None:
            interval = self.deepseek_sparse_attention_interval
            self.layer_types = [
                LINEAR_ATTENTION if (i + 1) % interval else SPARSE_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        unsupported = sorted(
            set(self.layer_types) - {LINEAR_ATTENTION, SPARSE_ATTENTION}
        )
        if unsupported:
            raise ValueError(f"Unsupported qwen4_exp layer types: {unsupported}")
        if len(self.layer_types) < self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )

        if self.hc_count <= 1:
            raise ValueError(f"qwen4_exp requires hc_count > 1, got {self.hc_count}")

        indexer_fields = {
            name: getattr(self, name)
            for name in (
                "indexer_n_heads",
                "indexer_kv_heads",
                "indexer_head_dim",
                "indexer_budget",
                "indexer_compress_ratio",
            )
        }
        if any(value is not None for value in indexer_fields.values()):
            missing = sorted(k for k, v in indexer_fields.items() if v is None)
            if missing:
                raise ValueError(f"qwen4_exp QSA config is missing {missing}")
            if any(value <= 0 for value in indexer_fields.values()):
                raise ValueError(
                    f"qwen4_exp QSA config must be positive: {indexer_fields}"
                )
            if self.indexer_kv_heads != 1:
                # The indexer reads its key cache as (B, kv_heads, L, head_dim),
                # which only aliases the projection's (B, L, kv_heads * head_dim)
                # layout for a single head -- more would be silently transposed.
                raise ValueError(
                    "qwen4_exp QSA requires indexer_kv_heads=1, got "
                    f"{self.indexer_kv_heads}"
                )
            if self.indexer_budget % self.indexer_compress_ratio:
                raise ValueError(
                    f"indexer_budget ({self.indexer_budget}) must be divisible by "
                    f"indexer_compress_ratio ({self.indexer_compress_ratio})"
                )
            rotary_dim = int(
                self.head_dim * self.rope_parameters["partial_rotary_factor"]
            )
            if rotary_dim > self.indexer_head_dim:
                raise ValueError(
                    "qwen4_exp attention RoPE dimensions must fit the QSA index "
                    f"head: rotary_dim={rotary_dim}, "
                    f"indexer_head_dim={self.indexer_head_dim}"
                )

        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if self.ple_layer_ids:
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            if ngram_heads <= 0 or self.ple_embed_dim % ngram_heads != 0:
                raise ValueError(
                    "ple_embed_dim must be divisible by the number of n-gram heads: "
                    f"{self.ple_embed_dim} % {ngram_heads} != 0"
                )
            if any(
                self.layer_types[i - 1] != LINEAR_ATTENTION for i in self.ple_layer_ids
            ):
                raise ValueError(
                    "qwen4_exp PLE is only supported on linear_attention layers"
                )
            if self.eos_token_id is None:
                raise ValueError("eos_token_id is required when PLE layers are enabled")

    @property
    def ple_eos_token_id(self) -> int:
        eos = self.eos_token_id
        return int(eos[0] if isinstance(eos, list) else eos)

    @property
    def uses_indexer(self) -> bool:
        return self.indexer_n_heads is not None


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vocab_size: int = 248320
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        self.eos_token_id = resolve_qwen_eos_token_id(
            self.eos_token_id, self.text_config
        )
        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )

    @classmethod
    def from_dict(cls, params):
        # Deserialize the nested configs first, otherwise their dataclass
        # defaults win over the values in config.json.
        params = dict(params)
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=True
        )
        return cls(**_config_kwargs(cls, params))
