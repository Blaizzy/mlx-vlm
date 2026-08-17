# Backward-compatible re-exports for the refactored trainer package.
# New code should import directly from the modality subpackages:
#   mlx_vlm.trainer.vlm.*     – Vision-Language trainers & datasets
#   mlx_vlm.trainer.peft.*    – LoRA / DoRA / adapter utilities
#   mlx_vlm.trainer.core.*    – Generic training infrastructure
#   mlx_vlm.trainer.stt       – Speech-to-Text (placeholder)
#   mlx_vlm.trainer.tts       – Text-to-Speech (placeholder)
#   mlx_vlm.trainer.diffusion – Diffusion models (placeholder)

from .core import (
    Colors,
    count_parameters,
    get_learning_rate,
    get_module_by_name,
    grad_checkpoint,
    not_supported_for_training,
    print_trainable_parameters,
    save_adapter,
    set_module_by_name,
)
from .peft import (
    DoRAEmbedding,
    DoRALinear,
    LoRAEmbedding,
    LoRALinear,
    LoRaLayer,
    LoRASwitchLinear,
    apply_lora_layers,
    find_all_linear_names,
    freeze_model,
    get_peft_model,
    linear_to_lora_layers,
    load_adapters,
    replace_lora_with_linear,
    unfreeze_modules,
)
from .vlm import (
    ORPOTrainingArgs,
    PreferenceVisionDataset,
    TrainingArgs,
    VisionDataset,
    train,
    train_orpo,
)
