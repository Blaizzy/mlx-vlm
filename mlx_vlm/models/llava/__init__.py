"""

Provides an initialization module for the deep learning model package, which mainly focuses on setting up the foundational classes, methods, and data structures required for a multi-modular neural network. The key components and their functionalities introduced in this module are as follows:

- `LanguageModel`: A class that is responsible for the language modeling part of a multi-modal system. It includes methods for processing text input and generating corresponding embeddings or hidden states.

- `VisionModel`: This class deals with the vision aspect of the neural network. It contains methods to handle image inputs and produce visual features from them.

- `LlavaMultiModalProjector`: A specialized class that projects the feature representations from both language and vision modules into a common space, enabling interactions between the modalities.

- `ModelConfig`: A configuration class representing the combined setups needed to instantiate a fully functional model by specifying the configurations for text, vision, and optional parameters such as vocabulary size, feature selection strategies, etc.

- `Model`: This key class binds all the distinct parts such as the vision tower, language model, and multi-modal projector into a cohesive whole. It includes methods for input embedding, loading pre-trained weights, and initiating the forward pass through the neural network.

- Importation of necessary submodules and utility functions required for the initialization and configuration of models.

**NOTE**: The provided methods and class implementations make extensive use of specific data structures (e.g., arrays from various frameworks like NumPy or MXNet) and assume a certain level of understanding of deep learning concepts, such as attention mechanisms, embeddings, and transformer architectures.

Additionally, the module ensures extensibility and flexibility by enabling the loading of pre-trained models from a specified path or a Hugging Face repository, making it easy to bootstrap the model with learned weights for downstream tasks.
"""

from .llava import (
    LanguageModel,
    Model,
    ModelConfig,
    TextConfig,
    VisionConfig,
    VisionModel,
)
