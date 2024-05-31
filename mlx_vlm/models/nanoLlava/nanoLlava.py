"""

## Module `nanoLlava`
The `nanoLlava` module provides the implementation for a multi-modal deep learning architecture designed to process a combination of text and image inputs, facilitating rich interactions with language and vision models. This module contains all the necessary building blocks, including models, configurations, and processors.

### `ModelConfig`
A data class that represents the configuration for the entire multi-modal model, encapsulating text and vision configurations, as well as specific parameters for the multi-modal interactions.

### `ImageProcessor`
A class that extends `BaseImageProcessor`, providing image preprocessing functionalities, such as conversion to RGB, resizing, rescaling, and normalizing the image inputs before they are fed into the vision model.

### `LlavaMultiModalProjector`
A `nn.Module` that implements a projection layer to align vision features with text features in the common multi-modal space.

### `SigLipVisionTower`
A subclass of `nn.Module` that wraps around the `VisionModel` to provide vision capabilities as part of the multi-modal model stack.

### `Model`
This central class ties the vision and language components together. It is responsible for merging image and text features, propagating through language model layers, and generating predictions.

### `Model` Static Methods
- `from_pretrained`: A method to instantiate the multi-modal model using pretrained weights from a specified path or Hugging Face repository.

The entire module is designed to be used within certain constraints and standards, relying heavily on the underlying structures provided by the `mlx` and `transformers` libraries. It is assumed that users of this module would be familiar with these libraries and deep learning model handling in general.
"""

import glob
import inspect
import json
import re
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import to_numpy_array

from ..base import BaseImageProcessor
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    """
    A dataclass that holds configuration parameters for a multimodal model which incorporates both text and vision configurations.

    Attributes:
        text_config (TextConfig):
             Configuration parameters related to the text processing component of the model.
        vision_config (VisionConfig):
             Configuration parameters related to the vision processing component of the model.
        model_type (str):
             A string indicating the type of the multimodal model.
        auto_map (dict):
             A dictionary for automatically mapping configuration parameters.
        hidden_size (int):
             The size of the hidden layers within the model.
        mm_hidden_size (int):
             The size of the multimodal hidden layers.
        mm_vision_tower (str):
             Specifies the architecture of the vision tower within the multimodal framework.
        mm_projector_type (str):
             Defines the type of the projector in the multimodal architecture, defaulting to 'mlp2x_gelu'.
        ignore_index (int):
             The index value that indicates to ignore the loss whenever it is encountered, default value is -100.
        image_token_index (int):
             The index value for image tokens, default value is -200.
        vocab_size (int):
             The size of the vocabulary, with a default value of 151936.
        Class Methods:
        from_dict:
             A class method for instantiating a ModelConfig object from a dictionary of parameters.


    """

    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    auto_map: dict
    hidden_size: int
    mm_hidden_size: int
    mm_vision_tower: str
    mm_projector_type: str = "mlp2x_gelu"
    ignore_index: int = -100
    image_token_index: int = -200
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        """
        Converts a dictionary into an instance of the class.
        This class method takes a dictionary and unpacks its items, using
        only those that correspond to the parameters defined in the
        class's initializer. This allows for the creation of a class
        instance with attributes set by the dictionary's key-value pairs.

        Args:
            params (dict):
                 A dictionary where keys correspond to the
                class's initializer parameters and the values are
                the values to be set for those parameters.

        Returns:
            An instance of the cls, with its attributes initialized
            to the values provided in the `params` dict, filtered to
            include only keys that match the class's initializer parameters.

        Raises:
            TypeError:
                 If any key in `params` does not correspond to an
                initializer parameter, it will be ignored and not
                raise an error. However, missing required initializer
                parameters that are not provided in `params` will
                result in a TypeError.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class ImageProcessor(BaseImageProcessor):
    """
    A class that extends BaseImageProcessor to apply a transformation pipeline to images.
    The ImageProcessor class provides a method to preprocess a single image or a list
    of images through a series of transformations. These transformations convert
    images to RGB if needed, resize and rescale the pixel values, normalize the
    image with given mean and standard deviation, and adjust the image to the
    specified channel format.

    Attributes:
        size (tuple of int):
             The target size for the images after resizing.
        resample (int):
             The resampling filter to use when resizing images.
        data_format (str):
             The desired channel format of the output images.
        rescale_factor (float):
             The factor by which to scale the image pixel values.
        image_mean (float or tuple of floats):
             The mean value(s) to use for normalization.
        image_std (float or tuple of floats):
             The standard deviation(s) to use for normalization.

    Methods:
        preprocess(images):
            Transform a single image or a list of images through the processing pipeline.

    Parameters:
        images (Union[Image.Image, List[Image.Image]]):
             The image or list of images to preprocess.

    Returns:
        (List[numpy.ndarray]):
             The list of preprocessed images as numpy arrays.

    """

    def preprocess(self, images):
        """
        Processes a list of images through a pipeline of transformation functions.
        This method takes either a single PIL Image or a list of PIL Images and applies a series of transformations to prepare them for further processing or model input. The transformations include converting images to RGB, converting them to numpy arrays, resizing, rescaling, normalizing, and reordering the channel dimensions. It uses Python's assert statement to ensure that the input is a list if it's not a single image. The function uses functools.reduce to apply each transformation sequentially to the images.

        Args:
            images (Union[Image.Image, List[Image.Image]]):
                 A single PIL Image or a list of PIL Images to be processed.

        Returns:
            (List[np.ndarray]):
                 A list of preprocessed images converted into numpy arrays.

        Raises:
            AssertionError:
                 If the input 'images' is neither a PIL Image nor a list of PIL Images.

        """
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(
                resize,
                size=self.size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)

        return images


class LlavaMultiModalProjector(nn.Module):
    """
    This class defines a multimodal projector module that is part of the Llava architecture,
    using linear transformations and activation functions to project input from one modality
    to a space compatible with another modality.

    Attributes:
        linear_1 (nn.Linear):
             A linear transformation layer that takes the hidden size of the
            vision component from `config` and transforms it into the text component's hidden size.
        gelu (nn.GELU):
             A Gaussian Error Linear Unit activation function.
        linear_2 (nn.Linear):
             A second linear transformation layer that maps the text component's
            hidden size to itself, effectively transforming the activation within the text modality space.

    Args:
        config (ModelConfig):
             An instance of the ModelConfig class providing configurations
            for the vision and text components of the model.

    Methods:
        __call__(x:
             mx.array) -> mx.array: Defines the forwarding pass of the input tensor `x` through
            the multimodal projector. It applies a linear transformation followed by GELU activation and
            another linear transformation, returning the projected output.

    Args:
        x (mx.array):
             Input tensor representing features from one modality (e.g., vision).

    Returns:
        (mx.array):
             The resulting tensor after projecting the input to the text modality space.


    """

    def __init__(self, config: ModelConfig):
        """
        Initializes a new instance of a model component, which is part of a larger model architecture.
        The constructor takes a configuration object that specifies the hidden size (number of features) of
        the vision and text components of the model. It then sets up two linear transformation layers and
        a GELU activation layer. The first linear layer maps from vision feature space to text feature
        space, and the second linear layer operates within the text feature space. The GELU activation
        is used between the two linear layers.

        Args:
            config (ModelConfig):
                 A configuration object with `vision_config` and `text_config` attributes.
                `vision_config.hidden_size` should define the size of the input feature space related to vision,
                and `text_config.hidden_size` defines the size of the feature space for text. Both are expected
                to be integers representing the number of features in their respective domains.

        Raises:
            This constructor does not explicitly raise exceptions, but if the `config` object is missing
            required attributes or the hidden sizes are not specified correctly, it may result in errors during
            attribute access or layer construction.

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
        Performs a forward pass on an input array using two linear transformations separated by a GELU activation function.

        Args:
            x (mx.array):
                 The input array to be transformed.

        Returns:
            (mx.array):
                 The transformed output array after applying two linear layers with a GELU activation function in between.

        Raises:
            NotImplementedError:
                 If any of the class's linear layers or the GELU function is not implemented.

        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class SigLipVisionTower(nn.Module):
    """
    A PyTorch module that serves as a wrapper for a VisionModel, which is typically used for processing visual input.
    The `SigLipVisionTower` class inherits from PyTorch's `nn.Module` and encapsulates a VisionModel configured with a given set of parameters.

    Attributes:
        vision_tower (VisionModel):
             An instance of the VisionModel class that is initialized
            with the provided `config` during the construction of this class.

    Args:
        config (VisionConfig):
             A configuration object containing the parameters for
            initializing the VisionModel instance within the tower.

    Methods:
        __call__(self, x, output_hidden_states):
            When an instance of `SigLipVisionTower` is called as a function, it internally
            invokes the `__call__` method of the `vision_tower` attribute.

    Args:
        x (mx.array):
             The input data to be processed by the vision model;
            expected to be a multidimensional array typically representing image data.
        output_hidden_states (Optional[bool]):
             A flag that determines whether or not
            the hidden states should be output
            alongside the main output of the
            vision model. Defaults to None,
            which typically causes the default
            behavior of the vision model to
            apply.

    Returns:
        (mx.array):
             The output of the vision model typically after processing the
            input array `x`. This might include the raw features, predictions,
            or any other form of processed visual information depending on
            the specific implementation and configuration of the VisionModel.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of the VisionModel class.

        Args:
            config (VisionConfig):
                 An instance of the VisionConfig class that provides the necessary configuration parameters for the VisionModel.

        Raises:
            None

        """
        super().__init__()
        self.vision_tower = VisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        """
        Calls the model with an input array and an optional flag to output hidden states.

        Args:
            x (mx.array):
                 An array containing the input data to be passed through the model.
            output_hidden_states (Optional[bool]):
                 Flag to indicate whether to return the hidden states or
                not. Defaults to None and will not return hidden states unless specifically requested.

        Returns:
            (mx.array):
                 The output of the vision_tower function which is typically a transformed array
                based on the input array and model's parameters, possibly including hidden states if
                requested.

        """
        return self.vision_tower(x, output_hidden_states)


class Model(nn.Module):
    """
    A multimodal neural network model that integrates visual and textual information.
    The Model class inherits from `nn.Module` and is designed to work with both text and image inputs.
    This model is envisioned to be utilized in scenarios where joint understanding of visual and text data is paramount.
    Functionalities include embedding textual and visual inputs, multimodal input preparation,
    and generating model logits with an optional caching mechanism for intermediate computations.
    This class relies on a configuration object to define its behavior, and uses specific
    submodules, like `SigLipVisionTower`, `LanguageModel`, and `LlavaMultiModalProjector`
    for the tasks at hand. The class methods facilitate the retrieval of input embeddings,
    loading of pretrained models, sanitization of weights, and execution of the model forward pass.

    Attributes:
        model_type (str):
             The type of model as defined in the configuration.
        config (ModelConfig):
             The configuration object containing all model parameters.
        vision_tower (SigLipVisionTower):
             The vision module that processes image inputs.
        language_model (LanguageModel):
             The language model for processing text inputs.
        mm_projector (LlavaMultiModalProjector):
             The multimodal projector that integrates language
            and vision features.

    Methods:
        __init__(self, config:
             ModelConfig)
            Initializes the model with the provided configuration.
        get_input_embeddings(self, input_ids:
             Optional[mx.array]=None, pixel_values: Optional[mx.array]=None)
            Computes the input embeddings for either the text or the image inputs.
            _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids)
            Combines the image and text embeddings into a single multimodal representation.
        __call__(self, input_ids:
             mx.array, pixel_values: mx.array, mask: Optional[mx.array]=None, cache=None)
            Processes the text and image inputs to generate model logits, optionally utilizing
            a caching mechanism.
        from_pretrained(path_or_hf_repo:
             str)
            Loads a pre-trained model from the specified local path or Hugging Face repository.
            sanitize(self, weights)
            Sanitizes the loaded weights to ensure compatibility with the model architecture.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the components of a multi-modal model with configurable architecture.
        This method initializes a multi-modal model by setting up a vision tower, a language model, and a multi-modal projector with the given configurations. Each component is initialized based on the specific configuration provided for that component.

        Args:
            config (ModelCOnfig):
                 A configuration object containing the settings for the model. This object typically includes separate configurations for vision and text components of the model.

        """
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = SigLipVisionTower(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = LlavaMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        """
        Generates input embeddings for a multimodal model from either text or image inputs, or both.
        This function prepares the input embeddings required by a multimodal model by processing the provided text and image inputs.
        If only text input is given, it uses the model's language_model component to generate embeddings from the input_ids.
        If image data is provided via the pixel_values argument, it processes the images through the vision_tower to
        obtain image features, projects these features using mm_projector, and then combines them with the text embeddings.
        The final input embeddings are prepared by the _prepare_inputs_for_multimodal function before being returned.

        Args:
            input_ids (Optional[mx.array]):
                 Array of input IDs for text inputs. Defaults to None.
            pixel_values (Optional[mx.array]):
                 Array of pixel values for image inputs. Defaults to None.

        Returns:
            An array representing the final input embeddings for the multimodal model.

        Raises:
            AssertionError:
                 If the processed image features shape does not match the expected dimensions.

        """
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        *_, hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = hidden_state[-1].astype(pixel_values.dtype)
        assert image_features.shape[-2] == 729

        image_features = self.mm_projector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        """
        Generates a concatenated embedding tensor from provided input segment embeddings and image features suitable for a multimodal transformer model.
        This function prepares the inputs by inserting image embeddings at appropriate positions within the sequence of text embeddings. It assumes that image tokens in the input IDs correspond to actual images and matches them with the provided image features.

        Args:
            image_features (np.ndarray):
                 A batch of image feature tensors of shape (num_images, num_image_patches, embed_dim),
                representing the features associated with each image that the model should process.
            inputs_embeds (np.ndarray):
                 The embedding representation of the input text of shape (batch_size, sequence_length, embed_dim).
            input_ids (np.ndarray):
                 An array of tokenized input text IDs. The '<image>' tokens in this sequence
                will be replaced by actual image embeddings based on their positions.

        Returns:
            (np.ndarray):
                 A tensor of concatenated embeddings of shape
                (batch_size, (num_image_patches * num_images) + sequence_length, embed_dim),
                ready to serve as input to a multimodal model.

        Raises:
            ValueError:
                 If the number of image tokens in the input IDs does not match the number of image inputs provided.

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
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        """
        Calls the Language Model with the provided inputs.
        This method passes input IDs and pixel values to the model's respective embedding layers
        to obtain input embeddings. It then feeds these embeddings, along with an optional mask
        and cache, into the language model to obtain logits and an updated cache.

        Args:
            input_ids (mx.array):
                 Token IDs for input text.
            pixel_values (mx.array):
                 Pixel values for any image input.
            mask (Optional[mx.array], optional):
                 Attention mask to avoid performing attention on padding token indices. Defaults to None.
            cache (optional):
                 Contains past pre-computed hidden-states (key and values in the attention blocks).
                Can be used to speed up decoding. Defaults to None.

        Returns:
            (Tuple containing):
            - logits (mx.array):
                 The classification or sequence generation logits from the model.
            (- cache):
                 Updated cache after running through the model. Can be passed back to the model
                in subsequent calls for faster decoding.

        """
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        """
        Loads a pre-trained model from a specified filesystem path or a Hugging Face repository.
        This method initializes a model using the configuration and weights found at the given path or Hugging Face (HF) model repository. It supports automatic download of configuration files (*.json), model weights (*.safetensors), Python files (*.py), tokenizer models (e.g., 'tokenizer.model'), and other necessary files following a pattern. If the model to be loaded is not available locally at the specified path, it attempts to download from the HF repository.

        Args:
            path_or_hf_repo (str):
                 The local file path or the identifier of the Hugging Face repository where the model configuration and weights are stored.

        Returns:
            (nn.Module):
                 An initialized model with the configuration and pre-trained weights loaded.

        Raises:
            FileNotFoundError:
                 If no *.safetensor files are found in the directory corresponding to the provided path or if the directory itself does not exist.

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
            config = json.load(f)

        siglip_config = AutoConfig.from_pretrained(config["mm_vision_tower"])
        text_config = AutoConfig.from_pretrained(config["language_model"])
        siglip_config = siglip_config.to_dict()
        text_config = text_config.to_dict()
        config["vision_config"] = siglip_config["vision_config"]
        config["text_config"] = text_config

        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        weights = LanguageModel(model_config.text_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        """
        Sanitizes the keys of a dictionary containing model weights.
        This function is responsible for converting the keys of the dictionary containing model weights to the appropriate format required by the consuming model architecture. It achieves this by using regular expressions to identify patterns within the keys and reformats them accordingly.
        The key transformation logic includes:
        - Stripping the prefix 'model.vision_tower' when present and leaving the rest of the key intact.
        - Replacing keys starting with 'model.mm_projector.0' with 'mm_projector.linear_1.' followed by the last segment of the key.
        - Replacing keys starting with 'model.mm_projector.2' with 'mm_projector.linear_2.' followed by the last segment of the key.
        - Replacing keys starting with 'lm_head' with 'language_model.model.' followed by the remainder of the key.
        - If a key starts with 'model.' and is followed by 'embed_tokens', 'norm', or 'layers', it is replaced with 'language_model.' and the remaining segments of the key.
        Additionally, it handles two specific cases for the keys related to 'vision_tower.vision_tower.vision_model.head.attention' by substituting the tail end of these keys with 'in_proj.bias' or 'in_proj.weight' respectively, correcting the key names for in-projection matrices.

        Parameters:
            weights (dict):
                 A dictionary mapping keys (str) that indicate parameter names to their respective weight values (tensor).

        Returns:
            (dict):
                 A dictionary with reformatted keys according to the architecture's requirements.

        Raises:
            None

        """
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.vision_tower", k)
                else (
                    f"mm_projector.linear_1.{k.split('.')[-1]}"
                    if re.match(r"^model\.mm_projector\.0", k)
                    else (
                        f"mm_projector.linear_2.{k.split('.')[-1]}"
                        if re.match(r"^model\.mm_projector\.2", k)
                        else (
                            f"language_model.model.{k}"
                            if re.match(r"^lm_head", k)
                            else (
                                f"language_model.{k}"
                                if re.match(r"^model\.(embed_tokens|norm|layers)", k)
                                else k
                            )
                        )
                    )
                )
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"vision_tower.vision_tower.vision_model.head.attention.in_proj.bias"
                if re.match(
                    r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_bias",
                    k,
                )
                else (
                    f"vision_tower.vision_tower.vision_model.head.attention.in_proj.weight"
                    if re.match(
                        r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_weight",
                        k,
                    )
                    else k
                )
            ): v
            for k, v in weights.items()
        }

        return weights
