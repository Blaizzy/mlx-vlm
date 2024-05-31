"""

The `paligemma` module provides a multimodal architecture that combines vision and language models to process both textual and visual information. It is designed to be capable of handling complex tasks that involve understanding and processing data from both modalities.

The module is structured around several core classes:

1. `ModelConfig`: A dataclass that contains configuration information for the model, including parameters for both the text and vision models, as well as additional settings specific to multimodal interactions.

2. `PaliGemmaMultiModalProjector`: A neural network module responsible for projecting image features from vision model output into a space compatible with the language model.

3. `Model`: The main class that encapsulates the entire multimodal model. It includes references to both the `VisionModel` and `LanguageModel`, as well as the `PaliGemmaMultiModalProjector`. It defines methods for preprocessing inputs, performing inference (the forward pass), and loading pre-trained models.

Additionally, the module defines auxiliar
"""

import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    """
    A dataclass that encapsulates the configuration parameters for a model that integrates text and vision capabilities.

    Attributes:
        text_config (TextConfig):
             An instance of TextConfig that contains text model configuration parameters.
        vision_config (VisionConfig):
             An instance of VisionConfig that contains vision model configuration parameters.
        model_type (str):
             A string representing the type of the model.
        vocab_size (int):
             An integer indicating the size of the vocabulary utilized by the model.
        ignore_index (int):
             An integer used during training to ignore a particular index in the loss computation. Default value is -100.
        image_token_index (int):
             An integer indicating the index of the token representing images in the model's vocabulary. Default value is 257152.
        hidden_size (int):
             An integer representing the size of hidden layers in the model.
        pad_token_validator (int):
             An integer representing the token that is used to pad sequences to a consistent length. Default value is 0.
        Classmethods:
        from_dict (Dict[str, Any]):
             A class method that instantiates a ModelConfig object from a dictionary of parameters. The method only considers parameters that are valid for the ModelConfig constructor, ignoring any extraneous parameters that may be present in the dictionary.

    """

    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int
    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, params):
        """
        Creates a new instance of the class from a dictionary of parameters.
        This class method takes a dictionary parameter and constructs a new
        instance of the class using only the keys that are present as parameters
        in the class constructor. It uses inspection to filter out any extraneous
        dictionary keys that do not match with the constructor parameters, thus
        preventing any TypeError that may occur from unexpected keyword arguments.

        Args:
            params (dict):
                 The dictionary of parameters to instantiate the class.

        Returns:
            An instance of the class configured with the provided parameters,
            ignoring any keys that are not constructor parameters.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class PaliGemmaMultiModalProjector(nn.Module):
    """
    A PyTorch module for projecting multimodal features into a common space.
    This class is designed to project features from a modality with a certain hidden size into a
    fixed-sized projection, typically for the purposes of multimodal learning where features
    from various modalities such as vision and language need to be aligned in a shared space.

    Attributes:
        linear (nn.Linear):
             A linear transformation layer that maps input features to the
            projected space. The layer configuration is defined by the `vision_config` of the
            `ModelConfig` object passed during initialization.

    Args:
        config (ModelConfig):
             A configuration object that contains settings for the model,
            specifically `vision_config` which holds configurations for the vision components
            such as hidden size and projection dimension.

    Methods:
        __call__(x:
             mx.array) -> mx.array: Defines the computation performed at every
            call. Applies the linear transformation to input `x`, projecting it to the
            output space.

    Raises:
        TypeError:
             If the input type for `x` in the `__call__` method is not a supported
            array type.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes a new instance of the class with the specified configuration.

        Args:
            config (ModelConfig):
                 The configuration object containing the vision model settings including the hidden_size
                and projection_dim used to define the linear transformation layer.

        Raises:
            TypeError:
                 If the 'config' argument is not of type ModelConfig.

        """
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Performs a forward pass through the module using the given input.
        This method is a special method that allows the object to be called as a function, effectively making the
        module callable. It takes an input tensor `x`, applies a linear transformation (defined by the `linear` method
        of the module), and returns the transformed output tensor.

        Args:
            x (mx.array):
                 An input tensor that is expected to be a numeric array of appropriate shape,
                compatible with the linear transformation.

        Returns:
            (mx.array):
                 The output tensor after applying the linear transformation to the input tensor.

        Raises:
            TypeError:
                 If the input is not of the expected type `mx.array`.

        """
        output = self.linear(x)
        return output


class Model(nn.Module):
    """
    A multi-modal deep learning model combining vision and language processing capabilities.
    This class is a PyTorch nn.Module that encapsulates the models for vision and language tasks, with additional
    support for multi-modal interactions. It requires a ModelConfig object during initialization to set up the
    specific configurations for the vision and language components, as well as the overall multimodal interaction.

    Attributes:
        model_type:
             A string indicating the high-level model type/designation.
        config:
             An instance of ModelConfig containing the configurations for vision, text, and multi-modal parts of the model.
        vision_tower:
             A VisionModel instance as defined for processing visual inputs.
        language_model:
             A LanguageModel instance as defined for processing textual inputs.
        multi_modal_projector:
             A PaliGemmaMultiModalProjector instance for handling multi-modal interactions.

    Methods:
        __init__(config):
             Initializes the model with the given configuration.
        get_input_embeddings(input_ids, pixel_values, mask):
             Generates input embeddings, combining textual and visual input features for multi-modal processing.
            _prepare_inputs_for_multimodal(image_features, inputs_embeds, input_ids, attention.*
        mask):
             Prepares inputs for the multi-modal layer by combining and aligning features from textual and visual sources.
        __call__(input_ids, pixel_values, mask, cache):
             Defines the forward pass of the model by processing input data and returning the output logits and cache for further processing or generation tasks.
        from_pretrained(path_or_hf_repo):
             Initializes a model with pretrained weights from a given path or Hugging Face repository.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the multimodal model with separate vision and language components, as well as a multimodal projector.

        Args:
            config (ModelConfig):
                 A configuration object containing model hyperparameters and specifications for the vision model, language model, and multimodal projector.

        Attributes:
            model_type (str):
                 The type of model as specified in the configuration.
            config (ModelConfig):
                 The configuration object that holds model settings.
            vision_tower (VisionModel):
                 The vision model component configured with the vision-specific settings from `config.vision_config`.
            language_model (LanguageModel):
                 The language model component configured with the text-specific settings from `config.text_config`.
            multi_modal_projector (PaliGemmaMultiModalProjector):
                 A linear projector that maps the output of the vision model to a space compatible with the language model output based on `config`.

        """
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        """
        Generates input embeddings for a multimodal model by processing either text or image inputs or both.
        This method accepts optional input_ids and pixel_values to handle text and image inputs, respectively. If only text
        is provided, it returns the embeddings from the language model directly. If both text and image inputs are provided,
        it generates embeddings for each, then projects and merges them for multimodal processing.

        Args:
            input_ids (Optional[mx.array]):
                 An array of tokenized text inputs. Default is None.
            pixel_values (Optional[mx.array]):
                 An array of raw pixel values for image inputs. Default is None.
            mask (Optional[mx.array]):
                 An array representing the attention mask for the inputs. Default is None.

        Returns:
            (Tuple):
                 A tuple containing two elements:
            (- final_inputs_embeds):
                 The final embeddings ready for multimodal processing.
            (- final_attention_mask_4d):
                 The attention mask corresponding to the final inputs.

        Raises:
            ValueError:
                 Raised if both input_ids and pixel_values are not provided.

        Note:
            The embedding process includes a projection and merging step for handling multimodal inputs.

        """
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_state, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype),
            output_hidden_states=True,
        )

        image_features = hidden_state[None, :].astype(pixel_values.dtype)
        image_features = self.multi_modal_projector(image_features)

        final_inputs_embeds, final_attention_mask_4d = (
            self._prepare_inputs_for_multimodal(
                image_features, inputs_embeds, input_ids, mask
            )
        )
        return final_inputs_embeds, final_attention_mask_4d

    def _prepare_inputs_for_multimodal(
        self, image_features, inputs_embeds, input_ids, attention_mask
    ):
        """
        Combines the textual and visual embeddings by expanding and inserting them into an initial zero matrix. It also adjusts the attention mask for multimodal input. The method supports only a single image feature and assumes that the image tokens in the input_ids are marked with a specific image token index defined in the config.

        Args:
            image_features (numpy.ndarray):
                 A 3D array of shape (num_images, num_regions, feature_dim), containing the image feature representations.
            inputs_embeds (numpy.ndarray):
                 A 2D array of shape (batch_size, sequence_length * embed_dim), containing the embedded representation of the input tokens.
            input_ids (numpy.ndarray):
                 A 2D array of shape (batch_size, sequence_length), containing the token IDs of the inputs.
            attention_mask (numpy.ndarray):
                 A 2D array of size (batch_size, sequence_length), denoting which tokens should be paid attention to and which should not.

        Returns:
            (tuple):
                 Contains the following two elements:
            - final_embedding (MXNet NDArray):
                 The combined embedding of text and image features for each item in the batch.
            - final_attention_mask_4D (MXNet NDArray):
                 The final attention mask, adjusted for the multimodal input, in a 4D array format.

        """
        _, _, embed_dim = image_features.shape

        batch_size, sequence_length = input_ids.shape
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        final_embedding = np.zeros((batch_size, sequence_length, embed_dim))

        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.config.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.config.pad_token_id

        # expand masks to match embedding dimension
        text_mask_expanded = np.expand_dims(text_mask, -1).repeat(embed_dim, axis=-1)
        pad_mask_expanded = np.expand_dims(pad_mask, -1).repeat(embed_dim, axis=-1)

        # insert padding and text token embeddings
        final_embedding = np.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = np.where(
            pad_mask_expanded, np.zeros_like(final_embedding), final_embedding
        )

        # insert image embeddings - the image mask is always less or equal to the sentence in length
        image_mask_expanded = np.expand_dims(image_mask, -1).repeat(embed_dim, axis=-1)
        final_embedding[image_mask_expanded] = scaled_image_features.flatten()

        final_embedding = np.where(
            pad_mask_expanded, np.zeros_like(final_embedding), final_embedding
        )

        attention_mask_expanded_1 = np.expand_dims(attention_mask, 1)
        attention_mask_expanded_2 = np.expand_dims(attention_mask, 2)
        final_attention_mask_4d = attention_mask_expanded_1 * attention_mask_expanded_2
        final_attention_mask_4d = final_attention_mask_4d
        final_attention_mask_4d = np.expand_dims(final_attention_mask_4d, 1).repeat(
            self.config.text_config.num_key_value_heads, axis=1
        )
        final_embedding = mx.array(final_embedding)
        final_attention_mask_4d = mx.array(final_attention_mask_4d)
        return final_embedding, final_attention_mask_4d

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
    ):
        """
        Transforms the input data into embeddings, passes it through the language model, and returns logits and updated cache.

        Args:
            input_ids (mx.array):
                 An MXNet NDArray containing tokenized text input.
            pixel_values (mx.array):
                 An MXNet NDArray containing the image features/pixel values.
            mask (Optional[mx.array], optional):
                 An optional MXNet NDArray that serves as a mask for the inputs. Defaults to None.
            cache (Optional[mx.array], optional):
                 An MXNet NDArray containing cached activations of the preceding calls to enable faster inference. Defaults to None.

        Returns:
            (tuple):
                 A tuple containing two elements;
            - logits (mx.array):
                 An MXNet NDArray representing the output logits from the language model.
            - cache (mx.array):
                 An updated MXNet NDArray cache resulting from the current execution, for use in subsequent calls.

        """
        input_embeddings, final_attention_mask_4d = self.get_input_embeddings(
            input_ids, pixel_values, mask
        )

        logits, cache = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            mask=final_attention_mask_4d,
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        """
        Loads a pre-trained model from the given path or HuggingFace repository.
        This static method initializes a model from pre-trained weights located at the specified file
        system path or downloaded from a HiffinFace repository. It reads the configuration from a
        JSON file and uses it to construct the model architecture. The weights are loaded from
        'safetensors' files. If no such files are found, it raises a FileNotFoundError.

        Args:
            path_or_hf_repo (str):
                 A file system path or a HuggingFace repository identifier where
                the pre-trained model's configuration and weights are stored.

        Returns:
            (Model):
                 An instance of the model class with pre-trained weights loaded.

        Raises:
            FileNotFoundError:
                 If the specified path does not exist or no 'safetensors' file
                is found within the given path.

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
        model.load_weights(list(weights.items()))
        return model
