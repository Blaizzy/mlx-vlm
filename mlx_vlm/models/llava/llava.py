"""

# `llava` Module Overview

The `llava` module is designed for multimodal learning, which typically involves processing and relating information from different modalities (e.g., visual and textual). In the context of this module, it particularly refers to integrating language models with vision models to work on tasks that involve both text and image data. The module includes classes and configurations that encapsulate language and vision model construction, as well as methods for intermodal interaction.

## Classes

### `ModelConfig`
Encapsulates both text and vision configuration settings necessary to construct a fully fledged multimodal model. This configuration is used to initialize the main `Model` class within the module and govern the behavior of its constituents.

### `Model`
The central class within the `llava` module. It is responsible for integrating the vision and the language models, applying the multi-modal projector, and handling the operations required to merge features from the different modalities for the predictive tasks it is designed for.

### `LanguageModel`
Represents a language model component within the `llava` module. Instances of this model are utilized to handle text inputs and provide textual features that will be further merged with the vision features.

### `VisionModel`
Represents the vision model which is responsible for processing the image input. The processed visual features are then projected to match the dimensionality of the textual features and allow for integration within the multimodal setup.

### `LlavaMultiModalProjector`
Used within the main `Model` class to project image features into the textual feature space, enabling the merging of features from the two different modalities within the multimodal model.

## Configurations

### `TextConfig`
A dataclass that holds the configuration settings related to language modeling. This includes parameters like model type, hidden size, and the number of attention heads.

### `VisionConfig`
A dataclass holding the vision model-related settings. It captures configuration specifics such as the number of hidden layers, the size of input images, patch size, and others.

## Methods and Properties
Class methods of the `llava` module typically include initialization routines, those that handle the call operations of the neural networks (forward passes), and methods for loading pretrained models. There are also properties and utilities to manage the embedding layers, feature selection strategies, and sanitizing weights for consistency.

The details of individual methods and properties are encoded within the docstrings of their respective classes, ensuring that users can reference the specifics as needed.

## Usage Notes
This JSON does not include usage examples or how to work with the `llava` module in practice. For those details, users should consult the comprehensive documentation that accompanies the module or reach out to the maintainers. This overview is concerned only with the structure and components of the `llava` module.
"""

import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    """
    A dataclass to represent configuration for a multimodal model that includes both text and vision components.

    Attributes:
        text_config (TextConfig):
             Configuration parameters for text processing components.
        vision_config (VisionConfig):
             Configuration parameters for vision processing components.
        model_type (str):
             Type of the model to be configured.
        ignore_index (int):
             Index that specifies a target value that is ignored and does not contribute to the input gradient.
            Default is -100.
        image_token_index (int):
             Index to use for image tokens in the vocabulary.
            Default is 32000.
        vision_feature_select_strategy (str):
             Strategy to select features from vision model.
            Default is 'default'.
        vision_feature_layer (int):
             Specifies the layer number from the vision model to extract features from.
            A value of -2 indicates the second to last layer.
        vocab_size (int):
             Specifies the size of the vocabulary.
            Default is 32000.
        Class Methods:
        from_dict(cls, params):
             Creates an instance of `ModelConfig` from a dictionary by matching
            its keys with the named parameters of the class constructor.

    Args:
        params (dict):
             A dictionary with keys corresponding to the names of parameters
            defined in the class constructor.

    Returns:
        (ModelConfig):
             An instance of `ModelConfig` initialized with parameters from the provided
            dictionary, only including keys that match the constructor's parameters.

    """

    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        """
        Creates a new class instance from a dictionary of parameters.
        This method uses the class's constructor (cls) and inspects its parameters to initialize an instance.
        It filters the input dictionary to only include items that match the constructor's parameter names.
        This allows for a dynamic way of instantiating objects from dictionaries that may contain extra keys.

        Args:
            params (dict):
                 A dictionary where keys are the name of the parameters expected by the class constructor,
                and values are the corresponding values to be used for instantiation.

        Returns:
            An instance of the cls, with attributes initialized as per the provided parameters.

        Raises:
            TypeError:
                 If any of the keys in the dictionary do not match the constructor's parameter names.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LlavaMultiModalProjector(nn.Module):
    """
    A neural network module responsible for projecting multimodal inputs between different hidden state sizes.
    This class is a component of a larger model architecture and contains a pipeline of operations to transform input data.
    It takes advantage of the linear layers and a Gaussian Error Linear Unit (GELU) activation function for the projection.

    Attributes:
        linear_1 (nn.Linear):
             The first linear transformation layer, which maps the input from the vision hidden size to the text hidden size.
        gelu (nn.GELU):
             The activation function that introduces nonlinearity into the transformation process after the first linear layer.
        linear_2 (nn.Linear):
             The second linear transformation layer, which further transforms the data within the text hidden size dimension.

    Args:
        config (ModelConfig):
             An instance of the ModelConfig class containing the configuration parameters for the vision and text components of the model.

    Methods:
        __call__(self, x:
             mx.array) -> mx.array: Processes the input `x` through the linear_1 layer, applies GELU activation, and then forwards the result through the linear_2 layer to generate the projected output.

    Note:
        - The input `x` is assumed to be a MXNet array with dimensions appropriate for the configured hidden size of the vision component.
        - This class is designed to be a part of a PyTorch module (inheriting from nn.Module) and must be used within the context of a PyTorch model.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes a new instance of the transformation model with configured layers.
        This constructor method initializes an instance of the transformation model by
        creating two linear layers with a GELU activation function in between. It maps the
        hidden states from the vision part of the model to the text part using the
        specified input and output dimensions provided in the `ModelConfig` object.

        Args:
            config (ModelConfig):
                 Configuration object containing the hidden sizes for the
                vision and text components of the model.

        Raises:
            TypeError:
                 If the provided `config` argument is not of type `ModelConfig`.

        """
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Performs a forward pass of the module.
        This method takes an input tensor, processes it through two linear layers
        with a Gaussian Error Linear Unit (GELU) activation in between.

        Args:
            x (mx.array):
                 The input tensor to the module.

        Returns:
            (mx.array):
                 The output tensor after being processed by the two linear layers and GELU activation.

        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(nn.Module):
    """
    A class representing a multi-modal neural network model leveraging separate vision and language towers for processing image and text inputs, respectively.

    Attributes:
        config (ModelConfig):
             Configuration parameters for the model.
        vision_tower (VisionModel):
             The vision sub-model for processing image data.
        language_model (LanguageModel):
             The language sub-model for processing text data.
        multi_modal_projector (LlavaMultiModalProjector):
             A module to project multi-modal features into a common space.
        vision_feature_layer (int):
             Index of the layer from which to extract vision features.
        vision_feature_select_strategy (str):
             Strategy to use for selecting vision features.

    Methods:
        get_input_embeddings:
             Retrieves and merges input embeddings from both vision and language models based on input data.
        _merge_input_ids_with_image_features:
             Combines the embeddings of image features with text embeddings.
        __call__:
             Processes the inputs through the model to generate logits and cache information.s
        from_pretrained:
             Loads the model weights from a pre-trained model.

    Raises:
        ValueError:
             If an unexpected vision feature selection strategy is provided, or if the number of image tokens does not match the number of image inputs.


    """

    def __init__(self, config: ModelConfig):
        """
        Initializes a multimodal model with specific configurations for vision and language tasks.

        Args:
            config (ModelConfig):
                 The configuration object containing sub-configurations
                for vision and language components of the model. It should contain
                'vision_config' for vision-related parameters, 'text_config' for
                language-related parameters, as well as additional settings like
                'vision_feature_layer' and 'vision_feature_select_strategy' to further
                customize the vision component.
                The initialization process involves setting up a vision tower model, a language model,
                a multimodal projector, and the configurations for selecting and processing vision features.
                The vision tower and language model are constructed using the corresponding configurations
                provided within 'config'.

        """
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        """
        Fetches the input embeddings combining the language and vision models.
        This function retrieves input embeddings using a specified language model for text inputs and
        a vision model for image inputs. If only text input is provided (input_ids), it processes
        the text through the language model alone. If image input (pixel_values) is provided, it
        also extracts hidden states from the desired layer of the vision model, applies a feature
        selection strategy, projects the image features for compatibility with language features,
        and merges them with the text embeddings.

        Args:
            self:
                 A reference to the instance of the class that contains the language and vision
                models, the selected feature layer, the feature selection strategy, and the
                multi-modal projector method.
            input_ids (Optional[mx.array]):
                 Tokenized text inputs meant for the language model.
            pixel_values (Optional[mx.array]):
                 Preprocessed image pixel values meant for the
                vision model.

        Returns:
            An MXNet array representing the merged input embeddings from the language and
            vision models, shaped appropriately for input to subsequent model layers.

        Raises:
            ValueError:
                 If an unrecognized vision feature selection strategy is provided.

        """
        if pixel_values is None:
            return self.language_model(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                "Unexpected feature selection strategy: "
                f"{self.vision_feature_select_strategy}"
            )

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        """
        Merges input embeddings with image features for a multimodal transformer model.
        Given a batch of input embeddings, input IDs, and image features from a model with
        a multimodal architecture, this function aims to generate the embeddings representing
        a merged sequence of text and image patches. It locates the positions of image tokens
        within the input IDs and segments the embeddings accordingly. Image features are then
        interleaved with these text embeddings segments to form a coherent sequence that can
        be fed into the transformer model. This sequence concatenates image patch embeddings
        immediately following their associated <image> token in the input sequence.

        Raises:
            ValueError:
                 If the number of <image> tokens found in the input_ids does not
                correspond to the number of image feature sets provided, indicating a mismatch
                between text and image data.
            Arguments:
            self:
                 A reference to the instance of the model class containing configuration
                details, specifically `self.config.image_token_index` representing the token index
                for image tokens in the vocabulary.
            image_features:
                 A tensor representing the extracted image features for all images
                in the batch. Expected shape is (num_images, num_image_patches, embed_dim), where
                `num_images` corresponds to the batch size, `num_image_patches` is the number of
                patches per image, and `embed_dim` is the embedding dimension size.
            inputs_embeds:
                 A tensor representing the input embeddings for the text. The shape
                is expected to be congruent with the size and structure of the input_ids, with an
                added embedding dimension.
            input_id:
                 A tensor representing the input IDs. This must be a sequence of integer
                indices into the model's vocabulary which includes special tokens denoting images.

        Returns:
            A tensor representing the concatenated embeddings of the text and the image patches,
            maintaining the original sequence order. The resulting tensor's shape is
            (1, num_image_patches*num_Sequence length of concatenated embeddings, embed_dim),
            ready for processing by the transformer model's encoder component.

        """
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

        if len(image_positions) != num_images:
            raise ValueError(
                f"The number of image tokens ({len(image_positions)}) does not "
                f" match the number of image inputs ({num_images})."
            )

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None
    ):
        """
        Generates output logits from the model given various inputs.
        This method is responsible for the forward pass of the model, producing logits that reflect the likelihood of various outputs given the input data. It can also make use of a cache for more efficient computation across multiple invocations when processing sequences incrementally.

        Args:
            input_ids (mx.array):
                 An array of token indices corresponding to the input text.
            pixel_values (mx.array):
                 An array containing the pixel values of an image associated with the text, if applicable.
            mask (mx.array):
                 An array indicating which elements of the input_ids should be attended to and which should be ignored.
            cache (Optional):
                 A cache object to store and retrieve intermediate states for efficient computation across multiple calls to the model.

        Returns:
            (A tuple containing the following elements):
            - logits (mx.array):
                 The output logits from the model, representing the unnormalized probabilities for different tokens/classes.
            (- cache):
                 The updated cache object if caching is enabled, otherwise the initial or unchanged cache provided as input.

        Raises:
            No explicit exceptions are raised by this method.

        """
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        """
        Loads a pre-trained model from the given file path or a Hugging Face repository.
        This static method initializes and returns a pre-trained model, given a local file path or a Hugging Face repository URL. It first checks for the existence of the provided `path_or_hf_repo`. If this is not found locally, it attempts to download the models from the Hugging Face repository specified in `path_or_hf_repo`, filtering for files with specific extensions. It then reads the model configuration from a 'config.json' file, constructs a model object using this configuration, and loads the model weights from files with the '.safetensors' extension. Finally, it returns the loaded model.

        Args:
            path_or_hf_repo (str):
                 The file path to the local directory containing the model files, or the identifier of the Hugging Face repository (e.g., 'username/model_name').

        Returns:
            (Model):
                 The loaded pre-trained model.

        Raises:
            FileNotFoundError:
                 If no files with the required '.safetensors' extension are found in the specified path.

        """
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model_config = ModelConfig.from_dict(model_config)

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config.text_config)

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = VisionModel.sanitize(weights)
        weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model
