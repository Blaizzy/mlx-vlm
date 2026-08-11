from typing import Any, Dict, Optional


class MuseGlimmerAssistantConfig:
    """Config for the Muse-Glimmer DFlash drafter (``MuseGlimmerAssistantModel``).

    Mirrors the published HF ``muse_glimmer_assistant`` config: a small stack of
    sliding-window decoder layers (5 layers, hidden 6656, GQA 32/8, head_dim 128)
    that denoise a block of ``block_size`` positions (an anchor token followed by
    ``block_size - 1`` mask tokens) conditioned on the target model's aux hidden
    states captured at ``target_layer_ids``.
    """

    model_type = "muse_glimmer_assistant"

    def __init__(
        self,
        hidden_size: int = 6656,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 19968,
        rms_norm_eps: float = 1e-05,
        sliding_window: int = 2048,
        block_size: int = 16,
        mask_token_id: int = 201818,
        target_layer_ids=None,
        max_position_embeddings: int = 131072,
        rope_theta: float = 500000.0,
        layer_types: Optional[list] = None,
        **kwargs: Any,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.target_layer_ids = (
            list(target_layer_ids)
            if target_layer_ids is not None
            else [1, 13, 25, 37, 49]
        )
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.layer_types = layer_types or ["sliding_attention"] * num_hidden_layers

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MuseGlimmerAssistantConfig":
        rope_params = config.get("rope_parameters", {}) or {}
        return cls(
            hidden_size=config.get("hidden_size", 6656),
            num_hidden_layers=config.get("num_hidden_layers", 5),
            num_attention_heads=config.get("num_attention_heads", 32),
            num_key_value_heads=config.get("num_key_value_heads", 8),
            head_dim=config.get("head_dim", 128),
            intermediate_size=config.get("intermediate_size", 19968),
            rms_norm_eps=config.get("rms_norm_eps", 1e-05),
            sliding_window=config.get("sliding_window", 2048),
            block_size=config.get("block_size", 16),
            mask_token_id=config.get("mask_token_id", 201818),
            target_layer_ids=config.get("target_layer_ids", [1, 13, 25, 37, 49]),
            max_position_embeddings=config.get("max_position_embeddings", 131072),
            rope_theta=rope_params.get("rope_theta", 500000.0),
            layer_types=config.get("layer_types"),
        )

    def validate(self) -> None:
        if len(self.target_layer_ids) < 1:
            raise ValueError(
                "Muse-Glimmer drafter requires at least one target aux layer id, "
                f"got {self.target_layer_ids}"
            )
        if self.block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {self.block_size}")


ModelConfig = MuseGlimmerAssistantConfig
