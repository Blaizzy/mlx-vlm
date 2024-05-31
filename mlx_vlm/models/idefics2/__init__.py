"""

Provides an initialization module for the `idefics2` deep learning framework, which includes
the necessary classes and functions to instantiate models with various configurations. The module comprises
several core classes such as `LanguageModel`, `Model`, `ModelConfig`, `PerceiverConfig`, `TextConfig`,
`VisionConfig`, and `VisionModel`, which are essential to set up the architecture of the neural networks
for processing multimodal inputs (text and images).
"""

from .idefics2 import (
    LanguageModel,
    Model,
    ModelConfig,
    PerceiverConfig,
    TextConfig,
    VisionConfig,
    VisionModel,
)
