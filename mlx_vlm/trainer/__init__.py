"""MLX-VLM training backend.

The notebook ``Trainer`` API and CLI share the engine in ``common``. Reusable
loss math and raw dataset loading live in ``losses`` and ``datasets``. The
``vlm`` package owns preprocessing and objectives. Existing functional recipes
remain available through ``train``/``train_orpo`` and their object facades.
"""

from .api import Trainer
from .common.callbacks import TrainingCallback, WandBCallback
from .common.config import CoreTrainingArgs
from .common.model import prepare_model_for_training
from .common.task import TrainingTask
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
    PreferenceVisionDataset,
    SFTTrainer,
    TrainingArgs,
    VisionDataset,
    train,
    train_orpo,
)
from .vlm.config import VLMTrainingArgs
from .vlm.dpo.config import DPOTrainingArgs
from .vlm.orpo.config import ORPOTrainingArgs
