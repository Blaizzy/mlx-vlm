# Copyright © 2024 MLX-VLM

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from .trainer import TrainingArgs
from .utils import grad_checkpoint, Colors, get_learning_rate

@dataclass
class OrpoTrainingArgs(TrainingArgs):
    beta: float = field(default=0.9, metadata={"help": "Exponential moving average parameter for reward normalization"})