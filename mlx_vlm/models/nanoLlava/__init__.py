"""

The `__init__` module of the `nanoLlava` package serves as the entry point for importing key components necessary for building and interacting with a multimodal (text and vision) machine learning model. The components imported through this module are foundational to configuring, processing, and executing tasks with the model.

The key components imported through this module include:

- `ImageProcessor`: A class designed to handle the preprocessing of images before they are provided to the model. It applies various transformations, such as converting images to RGB, resizing, rescaling, and normalizing them according to specific parameters.

- `LanguageModel`: This class encapsulates the functionality related to natural language processing, specifically text embedding and language understanding.

- `Model`: The central neural network model class that combines both vision and language understanding capabilities. It initializes separate components for image and text processing and integrates them using a multimodal projector.

- `ModelConfig`: A configuration class that holds the hyperparameters and configuration settings used to initialize the `Model` class.

- `TextConfig`: A configuration class specific to the text components of the model.

- `VisionConfig`: A configuration class specific to the vision components of the model.

- `VisionModel`: A class representing the vision tower of the multimodal model, responsible for processing image data.

By importing these components, users can configure and create instances of the model, process input data appropriately, and leverage the model for tasks involving both vision and language understanding.
"""

from .nanoLlava import (
    ImageProcessor,
    LanguageModel,
    Model,
    ModelConfig,
    TextConfig,
    VisionConfig,
    VisionModel,
)
