from dataclasses import dataclass

from ..deepseek_v3.config import ModelConfig as DeepseekV3Config


@dataclass
class ModelConfig(DeepseekV3Config):
    model_type: str = "xing4_0"
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    mhc_h_res_clamp_min: float = -30.0
    mhc_h_res_clamp_max: float = 30.0
