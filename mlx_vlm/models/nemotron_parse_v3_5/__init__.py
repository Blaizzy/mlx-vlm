# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import mlx_vlm.models.nemotron_parse_v3_5.processing_nemotron_parse_v3_5  # noqa: F401

from .config import ModelConfig, TextConfig, VisionConfig
from .nemotron_parse_v3_5 import LanguageModel, Model, VisionModel
