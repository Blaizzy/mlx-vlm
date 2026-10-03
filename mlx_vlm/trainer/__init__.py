"""MLX-VLM training backend.

Shared execution code lives next to this module. Reusable loss math and dataset
shapes are in ``losses`` and ``datasets``. Modality packages own their training
algorithms, preprocessing, and recipes.
"""

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
    LoRaLayer,
    LoRALinear,
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
    ORPOTrainer,
    ORPOTrainingArgs,
    PreferenceVisionDataset,
    SFTTrainer,
    TrainingArgs,
    VisionDataset,
    train,
    train_orpo,
)
