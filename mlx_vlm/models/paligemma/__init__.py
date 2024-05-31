"""

The `__init__` module initializes the core components required to build a multimodal deep learning model. The module imports several key classes and functions that are used in the creation of the model, namely `LanguageModel`, `Model`, `ModelConfig`, `TextConfig`, `VisionConfig`, `VisionModel`. Each of these components plays a specific role in the construction and configuration of the model that can process both text and image data.

- `LanguageModel`: Responsible for processing the text data. It handles the embedding and transformation of text inputs into an appropriate format for the multimodal interactions.

- `Model`: The primary class that encapsulates the complete model. It integrates both the language and vision components, manages multimodal input processing, and defines the forward pass of the model.

- `ModelConfig`: Contains configuration parameters for the model such as vocab size, token indices, and hidden sizes. It is used to instantiate the `Model` with the correct settings.

- `TextConfig`: Holds the configurations specific to the text processing part of the model such as vocabulary size and embeddings.

- `VisionConfig`: Contains settings for the vision processing part, detailing the model's vision architecture and parameters.

- `VisionModel`: A class dedicated to handling image data by extracting visual features that are later integrated with text features in the `Model`.

Overall, the `__init__` module is instrumental in setting up the Model's structure, by piecing together the individual elements required for processing and learning from multimodal data.
"""

from .paligemma import (
    LanguageModel,
    Model,
    ModelConfig,
    TextConfig,
    VisionConfig,
    VisionModel,
)
