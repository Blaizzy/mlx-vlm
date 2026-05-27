from __future__ import annotations

import mlx.core as mx


class ModelConfig:
    precision: mx.Dtype = mx.bfloat16
