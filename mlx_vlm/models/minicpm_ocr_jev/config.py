from dataclasses import dataclass

from ..qwen3_5.config import ModelConfig as QwenConfig


@dataclass
class ModelConfig(QwenConfig):
    model_type: str = "minicpm_ocr_jev"
    num_labels: int = 2
    pad_token_id: int = 248044
    problem_type: str = "single_label_classification"

    def __post_init__(self):
        super().__post_init__()
        if self.num_labels != 2:
            raise ValueError("MiniCPM-OCR Jev requires num_labels=2.")
