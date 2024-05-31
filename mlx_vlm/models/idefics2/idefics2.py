"""

## Module Overview: idefics2

This module provides an implementation of the Idefics2 model, which appears to be a multi-modal deep learning architecture capable of handling both text and image inputs. It comprises interconnected components designed to ingest, process, and combine language and visual elements for tasks that require understanding of both forms of data.

### Major Components

- **VisionConfig**: A configuration class for setting up the vision component of the model.
- **VisionModel**: A subclass of `nn.Module` tailored for the vision component, supporting image data processing.
- **TextConfig**: Similar to `VisionConfig`, but for configuring the language model component.
- **LanguageModel**: A subclass of `nn.Module` optimized for text data processing.
- **PerceiverConfig**: Configuration for the Perceiver-like mechanism of the model which is designed to aggregate and process information from different modalities.
- **ModelConfig**: Aggregates the configuration for the text, vision, and perceiver components of the model.

### Resampling and Attention Mechanism

- **Idefics2PerceiverAttention**: Implements the attention mechanism which combines latent queries with keys and values derived from the input.
- **Idefics2PerceiverLayer**: Represents a single Perceiver layer that processes the inputs using attention and feedforward neural network (MLP).
- **Idefics2PerceiverResampler**: A series of Perceiver layers that act as a resampling mechanism to integrate information over multiple iterations.

### Supporting Components

- **MLP**: A multilayer perceptron that is used as a part of other components for transformations and projections.
- **Idefics2Connector**: Facilitates the integration between the language and vision models.
- **Model**: The main class integrating the vision and language models with the connection mechanisms to form the complete multi-modal architecture.

### Methods and Model Initialization

- **Model.from_pretrained**: A method to load a pre-trained version of the Idefics2 model using weights from a specified local path or a HuggingFace repository.
- **Model.sanitize**: A utility method used to clean and prepare model weights for integration and consistency across the different modules.

### Configurations and Dataclasses

This module extensively uses Python's `dataclass` to manage configurations, making it straightforward to instantiate models with different settings. Each component has a corresponding configuration class, which utilizes class factories to instantiate configurations from dictionaries, providing flexibility for dynamic instantiation.

Each class and method in this module is implemented with careful attention to proper parameterization, allowing for complex orchestrations within the model as it processes distinct types of data through its subcomponents. This modular design facilitates the extension or modification of individual model components as needed.
"""

import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class PerceiverConfig:
    """
    A configuration class for the Perceiver model that sets up the architecture's parameters.

    Attributes:
        model_type (str):
             The type of Perceiver model.
        num_key_value_heads (int):
             The number of heads in key/value operations. Defaults to 4.
        resampler_depth (int):
             The depth of the resampling layer. Defaults to 3.
        resampler_head_dim (int):
             The dimension of each head in the resampling layer. Defaults to 96.
        resampler_n_heads (int):
             The number of heads in the resampling layer. Defaults to 16.
        resampler_n_latents (int):
             The number of latent variables in the resampling layer. Defaults to 64.
        Class Methods:
        from_dict (cls):
             Creates an instance of PerceiverConfig from a dictionary. It filters the dictionary keys to match the constructor arguments of the class.

    Args:
        params (dict):
             A dictionary containing parameters for configuring the Perceiver model.

    Returns:
        (PerceiverConfig):
             An instance of PerceiverConfig with the provided configuration parameters.

    """

    model_type: str
    num_key_value_heads: int = 4
    resampler_depth: int = 3
    resampler_head_dim: int = 96
    resampler_n_heads: int = 16
    resampler_n_latents: int = 64

    @classmethod
    def from_dict(cls, params):
        """
        Creates an instance of the class from a dictionary of parameters.
        The 'from_dict' class method takes a dictionary, filters its items to match those that are accepted by the class constructor (based on the constructor's signature), and creates a new instance of the class using the remaining parameters.

        Args:
            params (dict):
                 A dictionary containing potential parameters for the class constructor. Keys should correspond to the constructor's parameter names.

        Returns:
            An instance of the class, constructed with the allowed parameters from the provided dictionary.

        Raises:
            TypeError:
                 If any of the keys in 'params' do not correspond to the constructor's parameters, or if the instantiation fails due to incompatible or missing arguments.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    """
    A dataclass that represents the configuration of a model, which comprises various sub-configurations, a type, and additional parameters.

    Attributes:
        text_config (TextConfig):
             Configuration specific to text processing aspects of the model.
        vision_config (VisionConfig):
             Configuration specific to vision processing aspects of the model.
        perceiver_config (PerceiverConfig):
             Configuration specific to the Perceiver architecture aspects of the model.
        model_type (str):
             A string representing the type of model.
        ignore_index (int):
             The index that specifies a token to ignore during loss calculation, typically used in language models. Defaults to -100.
        image_token_index (int):
             The index that specifies the start of image token indices. Defaults to 32001.
        vocab_size (int):
             The size of the vocabulary used by the model. Defaults to 151936.
        Class Methods:
        from_dict(cls, params:
             dict) -> 'ModelConfig': Creates an instance of `ModelConfig` from a dictionary by unpacking parameters that match the dataclass's attributes.

    Args:
        params (dict):
             A dictionary where keys correspond to the attributes of the `ModelConfig` class.

    Returns:
        (ModelConfig):
             An instance of the ModelConfig class initialized with the parameters from the dictionary.

    """

    text_config: TextConfig
    vision_config: VisionConfig
    perceiver_config: PerceiverConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 32001
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        """
        Constructs an instance of the class from a dictionary of parameters.
        This class method takes a dictionary `params`, filters out keys that are not
        parameters of the class constructor (based on inspection), and then creates
        an instance of the class with the remaining parameters.

        Args:
            cls (type):
                 The class on which this method is called.
            params (dict):
                 A dictionary of parameters to be used for constructing an
                instance of the class. The dictionary may contain more items than needed.
                Only items that correspond to the class constructor's parameters will be used.

        Returns:
            An instance of `cls` constructed with the filtered parameters from `params`.

        Raises:
            KeyError:
                 If the `params` dictionary does not contain required parameters
                for creating an instance of the class.

        Note:
            This method relies on the `inspect` module to match the dictionary keys with
            the class constructor's parameter names.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Idefics2PerceiverAttention(nn.Module):
    """
    A custom PyTorn Idefics2 Perceiver attention module, designed for processing sequences by attending to key and value vectors with multiple heads.
    This module is initialized with a configuration object that specifies various hyperparameters for the attention mechanism.
    It creates linear projection layers for queries (q_proj), keys (k_proj), values (v_proj), and outputs (o_proj) based on the configuration.

    Args:
        config (ModelConfig):
             A configuration object containing hyperparameters.

    Attributes:
        n_heads (int):
             The number of attention heads for quering.
        n_kv_heads (int):
             The number of attention heads for key-value pairs.
        scale (float):
             A scaling factor for the attention scores, derived from head dimensionality.
        q_proj (nn.Linear):
             Linear projection layer for queries.
        k_proj (nn.Linear):
             Linear projection layer for keys.
        v_proj (nn.Linear):
             Linear projection layer for values.
        o_proj (nn.Linear):
             Linear projection layer to project the final output back to the original dimension.
            The '__call__' method computes the attention mechanism over the input with option to use cached key, value pairs for incremental updates.

    Args:
        x (mx.array):
             The query sequence tensor of shape (Batch, Seq_length, Dimension).
        kv (mx.array):
             The key/value sequence tensor to attend to.
        mask (Optional[mx.array]):
             An optional mask tensor to prevent attention to certain positions.
        cache (Optional[Tuple[mx.array, mx.array]]):
             An optional tuple containing cached key and value tensors from previous attention calculations.

    Returns:
        (mx.array):
             The output tensor after attention computations.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes a new instance of the model's layer with parameters for attention mechanisms.
        This constructor method sets up the projection layers for queries, keys, and values, along with an output projection layer
        based on the provided configuration settings. It calculates the scaling factor used in attention based on the
        dimension of attention heads.

        Args:
            config (ModelConfig):
                 An instance of ModelConfig containing various configuration settings for the model.

        Raises:
            ValueError:
                 If the configuration is incompatible or missing necessary information.

        """
        super().__init__()

        dim = config.text_config.hidden_size
        self.n_heads = n_heads = config.perceiver_config.resampler_n_heads
        self.n_kv_heads = n_kv_heads = config.perceiver_config.num_key_value_heads

        head_dim = config.perceiver_config.resampler_head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        kv: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Performs forward pass of the associated Attention mechanism with optional caching capability.
        This method takes input, key, and value tensors, projects them to queries, keys, and values for the attention mechanism,
        and computes the output of the attention. It optionally utilizes a cache to store previous states, which is useful in
        incremental decoding scenarios (e.g., autoregressive generation), where previous computations are reused to improve efficiency.

        Args:
            x (mx.array):
                 The query tensor with shape (B, L, D), where B is the batch size, L is the sequence length, and D is the feature dimension.
            kv (mx.array):
                 The key/value tensor with shape (B, *, D), where * denotes any sequence length, typically including both past and current key/value pairs.
            mask (Optional[mx.array]):
                 Optional mask tensor with shape (B, L, L) or (B, L, kv_seq_len) used to mask out invalid positions during the attention operation.
            cache (Optional[Tuple[mx.array, mx.array]]):
                 Optional tuple containing caches for keys and values, with each cache having shape (B, L, D).

        Returns:
            (mx.array):
                 The output tensor after computing attention with shape (B, L, D).

        Raises:
            mx.MXNetError:
                 If there's an error in MXNet operations during the attention computation.

        """
        B, L, D = x.shape
        kv_seq_len = L + kv.shape[1]
        hidden_states = mx.concatenate([kv, x], axis=-2)

        queries = self.q_proj(x)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, kv_seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, kv_seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Idefics2PerceiverLayer(nn.Module):
    """
    A layer module for the Idefics2 Perceiver architecture.
    This layer is responsible for processing inputs using latent-context cross attention followed by a multilayer perceptron (MLP).
    It normalizes both the input latents and the input context before applying the attention mechanism, and again normalizes the latents after the attention step.
    An MLP is then used to further process the attention result before adding it back to the original input latents.

    Attributes:
        hidden_size (int):
             The size of the hidden layer representations.
        n_latents (int):
             The number of latents used by the resampler.
        depth (int):
             The depth of the resampler in the perceiver configuration.
        rms_norm_eps (float):
             The epsilon value for root mean square layer normalization.
        input_latents_norm (nn.RMSNorm):
             RMSNorm layer for input latents.
        input_context_norm (nn.RMSNorm):
             RMSNorm layer for input context.
        self_attn (Idefics2PerceiverAttention):
             The self-attention module for the layer.
        post_attention_layernorm (nn.RMSNorm):
             RMSNorm layer following the self-attention.
        mlp (MLP):
             A multilayer perceptron with input layer size, four times the hidden layer size, and output layer size equal to the input layer size.

    Args:
        config (ModelConfig):
             The configuration object containing model hyperparameters.

    """

    def __init__(self, config: ModelConfig):
        """
        Constructs the initializer of the model class.

        Args:
            config (ModelConfig):
                 An instance of Model'Config containing the configuration details for the model.

        Attributes:
            hidden_size (int):
                 Number of hidden units in the model, determined from the text configuration.
            n_latents (int):
                 Number of latent variables in the Perceiver IO resampler configuration.
            depth (int):
                 Number of resampler layers in the Perceiver IO configuration.
            rms_norm_eps (float):
                 Small float added to the variance to avoid dividing by zero during normalization.
            input_latents_norm (nn.RMSNorm):
                 Root-mean-square layer normalization applied to the latents of the model.
            input_context_norm (nn.RMSNorm):
                 Root-mean-square layer normalization applied to the context of the model.
            self_attn (Idefics2PerceiverAttention):
                 The self-attention module used in the model.
            post_attention_layernorm (nn.RMSNorm):
                 Root-mean-square layer normalization applied after the self-attention module.
            mlp (MLP):
                 A multi-layer perceptron module used in the model.

        """
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.input_latents_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = Idefics2PerceiverAttention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.mlp = MLP(self.hidden_size, self.hidden_size * 4, self.hidden_size)

    def __call__(
        self,
        x: mx.array,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Performs a forward pass through a custom layer of a neural network model incorporating self-attention mechanism, layer normalization, and a multi-layer perceptron (MLP).
        This method accepts inputs and hidden states, applies self-attention followed by a residual connection, layer normalization, and an MLP block. An optional mask can be provided to influence the self-attention mechanism. Typically used in the context of transformer-based architectures where self-attention is a key component.

        Args:
            x (mx.array):
                 The input tensor with shape (batch_size, seq_length, feature_size).
            hidden_states (mx.array):
                 The hidden states tensor from the previous layer of the network with the same shape as `x`.
            mask (Optional[mx.array]):
                 An optional mask tensor with shape (batch_size, seq_length) to apply to the self-attention mechanism (default is None).

        Returns:
            (mx.array):
                 The output tensor after applying self-attention, residual connections, layer normalization, and MLP with the same shape as `x`.

        Raises:
            TypeError:
                 If the mask is provided but has an improper shape or type.

        """
        latents = self.input_latents_norm(x)
        context = self.input_context_norm(hidden_states)

        latents = self.self_attn(latents, context, mask=mask)

        latents = x + latents
        r = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = r + latents
        return latents


class Idefics2PerceiverResampler(nn.Module):
    """
    A resampler module for the Idefics2 Perceiver architecture, built as a PyTorch nn.Module.
    The Idefics2PerceiverResampler is designed to process input data through
    a sequence of Perceiver layers to map it onto a fixed set of latent variables.
    It creates a fixed-size latent array which is updated iteratively by the Perceiver
    layers, eventually normalized by an RMSNorm layer.

    Attributes:
        hidden_size (int):
             The size of the hidden layers as defined in the text configuration.
        n_latents (int):
             The number of latents to sample as defined in the Perceiver configuration.
        latents (mx.array):
             A matrix of ones initialized to hold the latent variables.
        layers (List[Idefics2PerceiverLayer]):
             A list of Perceiver layers to update the latents.
        norm (nn.RMSNorm):
             The RMS normalization layer applied to the output of the last Perceiver layer.

    Args:
        config (ModelConfig):
             The model configuration object containing settings for text and Perceiver layers.

    Methods:
        __call__(self, x:
             mx.array, mask: Optional[mx.array]): Processes the input data through the resampler.
            Applies the Perceiver layers sequentially to the input data, using an optional mask,
            and normalizes the latent output using RMSNormalization.


    Args:
        x (mx.array):
             The input data array to be resampled.
        mask (Optional[mx.array]):
             An optional mask array to be applied during processing.

    Returns:
        (mx.array):
             The resampled and normalized latent representation.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the model with specified configurations.
        This constructor method initializes an instance of the model by setting up
        various components based on the configurations provided. It initializes the
        hidden size, the number of latents, the latent vectors, the Perceiver layers,
        and the normalization layer with Root Mean Square normalization.

        Args:
            config (ModelConfig):
                 An object containing configurations for the model.
                This includes settings for text processing, Perceiver model components,
                and normalization parameters.

        Raises:
            ValueError:
                 If any of the required configurations are not provided or
                are invalid in the context of the model's architectural requirements.

        Note:
            This method should be called with a valid `ModelConfig` object that
            provides the necessary settings. Improper configuration can lead to
            unexpected behavior or failures within the model.


        """
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents

        self.latents = mx.ones((self.n_latents, self.hidden_size))
        self.layers = [
            Idefics2PerceiverLayer(config)
            for _ in range(config.perceiver_config.resampler_depth)
        ]
        self.norm = nn.RMSNorm(self.hidden_size, eps=config.text_config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None):
        """
        Performs a forward pass on an input sequence using the transformer's layers.

        Args:
            x (mx.array):
                 The input sequence to the transformer, expects a tensor with shape [batch_size, seq_len, ...].
            mask (Optional[mx.array], optional):
                 An optional mask tensor with shape [batch_size, seq_len] where 0 values indicate positions that should be masked from attention. Defaults to None.

        Returns:
            (mx.array):
                 The output tensor of the transformer with the same batch dimension as x after being processed through the model's layers and normalization.

        Raises:
            TypeError:
                 If `x` is not an instance of mx.array.

        """
        h = mx.expand_dims(self.latents, axis=0)
        h = mx.repeat(h, x.shape[0], axis=0)

        for layer in self.layers:
            h = layer(h, x, mask=mask)

        return self.norm(h)


class MLP(nn.Module):
    """
    A Multilayer Perceptron (MLP) module implementing a simple neural network with one hidden layer.

    Attributes:
        gate_proj (nn.Linear):
             A linear transformation applied to the input without a bias term, projecting
            the input to the hidden dimension `hidden_dim`.
        down_proj (nn.Linear):
             A linear transformation, without bias, mapping from the hidden dimension
            to the output dimension `output_size`.
        up_proj (nn.Linear):
             Another linear transformation without bias applied to the input,
            projecting to the same hidden dimension `hidden_dim`.

    Args:
        dim (int):
             The size of the input dimension.
        hidden_dim (int):
             The size of the hidden layer.
        output_size (int):
             The size of the output dimension.
            Note that this class extends nn.Module and should be used with an input tensor `x`. The `__call__`
            method will apply a gated linear unit activation, represented by `nn.silu`, on the gate projection of
            the input, multiply it by the projection of the input on the upward path, and then pass the result
            to the downward projection to generate the final output.

    """

    def __init__(self, dim, hidden_dim, output_size):
        """
        Initialize the class with the specified layer dimensions and projections.

        Args:
            dim (int):
                 The size of input features.
            hidden_dim (int):
                 The size of hidden layer features.
            output_size (int):
                 The size of the output features.

        Attributes:
            gate_proj (nn.Linear):
                 The linear transformation layer for gating mechanism.
            down_proj (nn.Linear):
                 The linear transformation layer that projects the hidden dimensions to the output size.
            up_proj (nn.Linear):
                 The linear transformation layer that projects the input dimensions to the hidden dimensions.

        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_size, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Performs a forward pass on input data through the gate, up projection, and down projection operations.
        This method implements a custom computation that passes the input tensor 'x' through
        several operations. First, it applies a gating mechanism by utilizing a `gate_proj`
        function, followed by a non-linear activation function `nn.silu`. The result of
        this gating mechanism is then element-wise multiplied with the 'up_projection'
        of the input tensor. Finally, the result of this multiplication is passed through
        a 'down_projection', and the output tensor is returned.
        The `__call__` method is typically invoked when the instance is called like a
        function, allowing the object to be used in a functional programming style.
        Arguments:
        x (mx.array): The input tensor to the custom forward pass.

        Returns:
            (mx.array):
                 The output tensor after applying the gate projection, non-linear
                activation, up projection, and down projection operations.

        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Idefics2Connector(nn.Module):
    """
    A PyTorch module for establishing a connection between different modalities in a deep learning model, specifically designed to project one modality into another and pass the projected data through a resampling process using the Idefics2PerceiverResampler.
    The Idefics2Connector class uses a multilayer perceptron (MLP) to project from the hidden size of the vision configuration to an intermediate size and then to the hidden size of the text configuration as specified in the given ModelConfig object. It then applies a resampling process through the Idefics2PerceiverResampler to the projected data.

    Attributes:
        modality_projection (MLP):
             A multilayer perceptron that projects the input data from vision modality size to text modality size.
        perceiver_resampler (Idefics2PerceiverResampler):
             The resampling component used to process the data after the modality projection.

    Args:
        config (ModelConfig):
             A configuration object containing the vision and text configuration parameters including hidden sizes and intermediate sizes.

    Methods:
        __call__(self, x:
             mx.array, mask=None) -> mx.array: Passes an input array through the modality projection and perceiver resampler, with an optional mask array.

    Args:
        x (mx.array):
             The input data array to be processed.
        mask (optional):
             An optional mask array, default is None, which can be used to specify which parts of the input are valid.

    Returns:
        (mx.array):
             The output array after processing through the projection and resampling stages.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the instance with a modality projection and a Perceiver resampler.

        Args:
            config (ModelConfig):
                 The configuration object containing various sub-configurations for the model. This includes configurations
                for vision and text modalities, as well as the Perceiver architecture.

        Raises:
            TypeError:
                 If the `config` is not an instance of `ModelConfig`.

        """
        super().__init__()
        self.modality_projection = MLP(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
        )

        self.perceiver_resampler = Idefics2PerceiverResampler(config)

    def __call__(self, x: mx.array, mask=None) -> mx.array:
        """
        Processes input data through the modality projection and perceiver resampler to generate an output.
        The method takes an input tensor `x`, applies a modality-specific projection to it, followed by a resampling through the perceiver resampler.
        An optional `mask` can be provided to affect the resampling process, which is useful for variable length or incomplete data.

        Args:
            x (mx.array):
                 The input tensor that needs to be processed.
            mask (Optional[any], default=None):
                 An optional mask to apply during the resampling.
                This can be used to inform the model about which parts of the input are valid and
                should be focused on during processing.

        Returns:
            (mx.array):
                 The processed output after applying the modality projection and
                the perceiver resampler to the input `x`.

        Raises:
            TypeError:
                 If the input types are not as expected.


        """
        x = self.modality_projection(x)
        x = self.perceiver_resampler(x, mask=mask)
        return x


class Model(nn.Module):
    """
    A PyTorch compatible neural network model that integrates vision and language processing components.
    The Model class combines a VisionModel for processing image data and a LanguageModel for text data, connecting
    these using an Idefics2Connector. It features methods for embedding multimodal inputs, loading pretrained weights,
    and performing forward propagation through the combined multimodal network.

    Attributes:
        model_type (str):
             Type of model as specified in the ModelConfig.
        config (ModelConfig):
             Configuration object containing model settings.
        vision_model (VisionModel):
             Initialized VisionModel object based on vision configuration.
        language_model (LanguageModel):
             Initialized LanguageModel object based on text configuration.
        connector (Idefics2Connector):
             Connector to facilitate interactions between vision and language components.

    Methods:
        get_input_embeddings:
             Produces a combined embedding from text and image inputs.
        _prepare_inputs_for_multimodal:
             Prepares and merges embeddings for multimodal inputs.
        __call__:
             Performs a forward pass of the model given text and image inputs.
        from_pretrained:
             Class method to instantiate a model with pretrained weights from a specified path or repository.
        sanitize:
             Cleans the key names in the state dictionary to match the model's architecture.
            The class provides functionality for handling multimodal (text and visual) data by employing a connector
            to link feature representations from separate modality-specific models. Furthermore, it supports loading
            pretrained components and includes methods for preprocessing and managing embeddings to facilitate
            multimodal interactions during inference.

    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the main model with configuration settings for vision and language models, as well as the connector module.

        Args:
            config (ModelConfig):
                 An instance of ModelConfig containing all necessary configurations for initializing the vision and language models, and the connector module.

        """
        self.model_type = config.model_type
        self.config = config

        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.connector = Idefics2Connector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
    ):
        """
        Generates the input embeddings for a multimodal model based on either provided text or image inputs.
        This function accepts input identifiers for text (input_ids), pixel values for images (pixel_values), and an optional pixel attention mask (pixel_attention_id) to create an embedding that the multimodal model can process. If only text is provided, it returns the embeddings from the language model component. For image inputs, it generates image features by processing the pixel values through the vision model component, and then combines these features with the text embeddings from the language model using a connector, finally preparing them for the multimodal model.

        Args:
            input_ids (Optional[mx.array]):
                 An array of input identifiers for the text. Defaults to None.
            pixel_values (Optional[mx.array]):
                 An array of pixel values for the image. Defaults to None.
            pixel_attention_mask (Optional[mx.array]):
                 An array representing which pixels should be attended to. Defaults to None.

        Returns:
            (mx.array):
                 The final input embeddings for the multimodal model, combining both text and image features if both are provided, or just the text features if only text inputs are provided.

        Raises:
            ValueError:
                 If none of the input types (text or image) are provided, or the input types are not in the expected array format.

        """
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        pooler_output, embeddings, hidden_state = self.vision_model(
            pixel_values[0].transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = pooler_output[None, :].astype(pixel_values.dtype)

        image_features = self.connector(image_features, mask=None)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        """
        Prepares and concatenates embeddings for text and image features for multimodal input processing.
        This function assumes a batch size of 1 and requires the image features, input embeddings, and input ids as inputs. It identifies the positions of the special <image> tokens within the input_ids and segments the input embeddings accordingly. The image features are then split into embeddings corresponding to individual images, which are interspersed with the segmented text embeddings to form a coherent sequence of embeddings. This sequence is returned after concatenating all embeddings along the sequence length dimension, effectively merging text and image information into a single tensor appropriate for processing by a multimodal model.

        Args:
            image_features (np.ndarray):
                 A numpy array with shape (num_images, num_image_patches, embed_dim) representing the image features.
            inputs_embeds (np.ndarray):
                 A numpy array representing input embeddings for the text, assuming a single batch.
            input_ids (np.ndarray):
                 A numpy array of token ids for the inputs, used to identify the positions of special <image> tokens.

        Returns:
            (np.ndarray):
                 A numpy array that concatenates image and text embeddings, with shape (1, num_image_patches*num_images + sequence_len, embed_dim).

        Raises:
            ValueError:
                 If the provided input_ids do not correspond to a batch size of 1, as this function is specifically designed to handle single-batch inputs.

        """
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()

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
        Generates the outputs of a language model given different input arrays and an optional cache state.

        Args:
            input_ids (mx.array):
                 A MXNet array of input token IDs.
            pixel_values (mx.array):
                 A MXNet array of pixel values associated with the input.
            mask (mx.array):
                 A MXNet array to mask certain input parts during processing.
            cache (optional):
                 A precomputed cache state for the language model to reuse calculated values, offering speed improvements.

        Returns:
            (A tuple containing):
            - logits (mx.array):
                 The prediction scores (logits) from the language model.
            (- cache):
                 Updated cache state after processing the input, which can be reused in subsequent calls.

        Raises:
            ValueError:
                 If any of the model inputs are invalid or incorrectly formatted.

        """
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits, cache

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        """
        Loads a pretrained model from the given path or Hugging Face repository.
        This static method initializes a model with the configuration and weights from a
        predefined path or a Hugging Face repository. It expects certain file patterns
        in the specified directory, including JSON configuration files, .safetensors
        files for weight parameters, and other necessary files such as tokenizers.
        If a path to a Hugging Face repository is given, the `snapshot_download`
        function will be used to download the repository contents, respecting
        include patterns for required file types.

        Args:
            path_or_hf_repo (str):
                 A local file system path or a Hugging Face repository
                identifier where the model's configuration and weights can be
                loaded from.

        Returns:
            (Model):
                 An instance of the Model class with weights loaded and ready
                for inference or further training.

        Raises:
            FileNotFoundError:
                 If no safetensors files are found in the specified
                path or repository, indicating that the model weights are
                missing.

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

        text_config = AutoConfig.from_pretrained(config["text_config"]["model_type"])
        text_config = text_config.to_dict()
        config["text_config"] = text_config
        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])
        model_config.perceiver_config = PerceiverConfig.from_dict(
            config["perceiver_config"]
        )

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
        Sanitizes the provided model parameter weight names by re-mapping them based on their original names.
        This function takes the model parameter weights as input and performs two key transformations. The first transformation removes the 'model.' prefix from parameter names that begin with it and ensures that they start with 'language_model.'. The second transformation specifically targets parameter names that start with 'lm_head.' and renames them to simply start with 'language_model.'. If the parameter names do not match these cases, they remain unchanged. This re-mapping is done to standardize model parameter names across different model architectures, making it easier to manage and integrate weights.

        Args:
            weights (dict):
                 A dictionary where keys are the parameter names and values are the weights
                associated with the parameter names. These weights could be tensors or any other data form
                that represents the parameters of a model.

        Returns:
            (dict):
                 A new dictionary where the keys are updated parameter names following the standardized
                naming convention and values are the corresponding weights as provided in the input dictionary.

        """
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.", k)
                else (f"language_model.{k}" if re.match(r"^lm_head\.", k) else k)
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"language_model.{k.split('.', 1)[1]}"
                if re.match(
                    r"^text_model\.",
                    k,
                )
                else k
            ): v
            for k, v in weights.items()
        }

        return weights
