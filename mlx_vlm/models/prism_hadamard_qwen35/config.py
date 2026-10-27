from dataclasses import dataclass, field

from ..qwen3_5.config import ModelConfig as Qwen3_5ModelConfig


@dataclass
class ModelConfig(Qwen3_5ModelConfig):
    schema_version: int = 2
    base_model_type: str = "qwen3_5"
    tensor_namespace: str = "mlx-vlm-qwen3_5"
    gdn_activation_layout: str = "grouped"
    modules: list[dict] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.schema_version != 2 or self.base_model_type != "qwen3_5":
            raise ValueError("Only schema 2 Qwen3.5 Hadamard packs are supported")
        if self.tensor_namespace != "mlx-vlm-qwen3_5":
            raise ValueError("Unsupported Hadamard tensor namespace")
        if self.gdn_activation_layout != "grouped":
            raise ValueError("Hadamard packs require grouped GDN activations")
        if self.quantization != {"bits": 2, "group_size": 128, "mode": "affine"}:
            raise ValueError(
                "Hadamard packs require 2-bit affine weights, group size 128"
            )
        if not self.modules:
            raise ValueError("Hadamard pack is missing its packed module manifest")
