"""

The vision module encapsulates the implementation of a vision encoder model similar to the architecture used in CLIP (Contrastive Language-Image Pretraining), specifically tailored for processing image data. It defines a series of classes and functions, organized as components of a deep neural network, that facilitate the construction and operation of such a model. The central elements provided by this module include data structures for configuring the model, building blocks for the neural network architecture, and utility functions for managing network parameters and array shapes.

### Key Components:

- `VisionConfig`: This is a dataclass used to define different hyperparameters and configurations required to build the vision model. It simplifies the process of instantiating a model with specific settings through the `from_dict` class method.

- `check_array_shape`: A function which takes an array as input and checks if it conforms to the required shape with specific constraints on the number of channels and kernel dimensions - necessary for verifying convolutional weights.

- `Attention`: A module representing the multi-head attention mechanism with customizable dimensions for queries, keys, values, and the outputs.

- `MLP`: This module implements a multi-layer perceptron with GELU activation. It is typically used as a feed-forward network within a transformer encoder block.

- `EncoderLayer`: Encapsulates a single layer of a transformer encoder, which includes self-attention, normalization, and a feed-forward network (MLP).

- `Encoder`: Assembles multiple `EncoderLayer` instances to form the encoder portion of the model.

- `VisionEmbeddings`: Handles the embedding of image pixels to a higher dimensional space suitable for processing by the encoder. It includes patch embeddings created through a convolutional layer and position embeddings.

- `ClipVisionModel`: Embodies the full vision model structure including the vision embeddings, pre-layer normalization, the encoder stack, and post-layer normalization.

- `VisionModel`: A wrapper class that brings together the architecture defined by `VisionConfig` and the `ClipVisionModel`. It serves as the entry point for process inputs through the network.

- `sanitize`: A method within `VisionModel` that is used to ensure the model's weights are in the correct format, especially after loading from external sources or checkpoints.

This module is intended for use within a larger ecosystem, likely involving image processing tasks such as feature extraction or model fine-tuning on vision-related tasks. Through the use of classes like `Encoder` and `VisionEmbeddings`, users can construct flexible and powerful models for a wide range of computer vision applications.
"""

import inspect
import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    """
    A dataclass representing the configuration parameters for a vision model.

    Attributes:
        model_type (str):
             Type of the vision model.
        num_hidden_layers (int):
             Number of hidden layers in the model. Defaults to 24.
        hidden_size (int):
             Size of the hidden layers. Defaults to 1024.
        intermediate_size (int):
             Size of the 'intermediate' layer in the transformer. Defaults to 4096.
        num_attention_heads (int):
             Number of attention heads in each attention layer. Defaults to 16.
        image_size (int):
             Size of the input image in pixels. Defaults to 336.
        patch_size (int):
             The size of each image patch. Defaults to 14.
        projection_dim (int):
             Dimension of the projection space. Defaults to 768.
        vocab_size (int):
             The size of the vocabulary. Defaults to 32000.
        num_channels (int):
             Number of image channels (e.g., 3 for RGB images). Defaults to 3.
        layer_norm_eps (float):
             Epsilon parameter for layer normalization. Defaults to 1e-05.
        Class Methods:
        from_dict(cls, params):
             Constructs a VisionConfig instance from a dictionary by filtering out any keys that are not
            recognized as configuration attributes. This ensures that attributes passed to the constructor
            match the ones defined in the dataclass. Any extra parameters in the dictionary are ignored.

    Args:
        params (dict):
             A dictionary where keys are names of the attributes and values are the values for those attributes.

    Returns:
        (VisionConfig):
             A new instance of VisionConfig initialized with the provided attribute values.

    """

    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 336
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, params):
        """
        Creates an instance of the class using the provided dictionary.
        This class method takes a dictionary `params` as input which should contain keys and values for initializing an instance of the class. It filters out any keys that are not present in the class constructor's signature, thereby ensuring any extraneous information in `params` is ignored.

        Args:
            params (dict):
                 A dictionary with keys corresponding to the class constructor's parameter names and their associated values.

        Returns:
            An instance of the class initialized with the values from the `params` dictionary.

        Raises:
            TypeError:
                 If any of the keys in `params` do not match the constructor's parameter names or if the values are not compatible with the constructor parameters.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    """
    Checks if a given 4D array adheres to specific dimensional constraints.
    The function requires the array to have exactly 4 dimensions. It verifies that out of the dimensions (out_channels, kH, kW), the number of out_channels is the largest and that the dimensions kH and kW are equal.

    Args:
        arr (ndarray):
             The array to be checked for compliance with shape constraints.

    Returns:
        (bool):
             True if the array shape matches the criteria, otherwise False.

    """
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class Attention(nn.Module):
    """
    A module for multi-head attention mechanism.
    This module encapsulates a multi-head attention mechanism, where multiple
    sets of query, key, and value weight matrices are applied to the input to
    compute the attention scores. It is commonly used in transformer models.

    Attributes:
        num_heads (int):
             The number of attention heads.
        scale (float):
             A scaling factor for the dot product of queries and keys.
        q_proj (nn.Linear):
             The linear layer for projecting queries.
        k_proj (nn.Linear):
             The linear layer for projecting keys.
        v_proj (nn.Linear):
             The linear layer for projecting values.
        out_proj (nn.Linear):
             The linear layer for projecting the final output.

    Args:
        dims (int):
             The total dimensionality of the query, key, and value vectors.
        num_heads (int):
             The number of parallel attention heads.
        query_input_dims (Optional[int]):
             The dimensionality of the input query matrix.
            If None, defaults to 'dims'.
        key_input_dims (Optional[int]):
             The dimensionality of the input key matrix.
            If None, defaults to 'dims'.
        value_input_dims (Optional[int]):
             The dimensionality of the input value matrix.
            If None, defaults to 'key_input_dims'.
        value_dims (Optional[int]):
             The internal dimensionality of the value vectors
            after projection. If None, defaults to 'dims'.
        value_output_dims (Optional[int]):
             The dimensionality of the output value matrix
            after attention has been applied. If None,
            defaults to 'dims'.
        bias (bool):
             Whether to add a bias term to the projection layers. Defaults to False.

    Raises:
        ValueError:
             If 'dims' is not divisible by 'num_heads'.
        Method:
        __call__:
             Computes the multi-head attention mechanism on input query,
            key, and value matrices, optionally applying a mask.

    Note:
        This class is designed to be used within a larger neural network architecture
        and assumes that the input matrices have already been properly prepared and
        batched.

    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        """
        Initializes a multi-head attention layer with optional separate dimensions for queries, keys, and values.
        This constructor sets up the projection layers for queries (q_proj), keys (k_proj), values (v_proj),
        and the output projection layer (out_proj). It also computes the scaling factor for the dot
        products in the attention mechanism based on the head dimensionality.

        Args:
            dims (int):
                 The number of feature dimensions for the model.
            num_heads (int):
                 The number of attention heads.
            query_input_dims (Optional[int]):
                 The dimensionality of the input queries. If `None`, defaults to `dims`.
            key_input_dims (Optional[int]):
                 The dimensionality of the input keys. If `None`, defaults to `dims`.
            value_input_dims (Optional[int]):
                 The dimensionality of the input values. If `None`, defaults to `key_input_dims`.
            value_dims (Optional[int]):
                 The dimensionality of the value projections. If `None`, defaults to `dims`.
            value_output_dims (Optional[int]):
                 The dimensionality of the output after combining the values. If `None`, defaults to `dims`.
            bias (bool):
                 Whether to include bias terms in the projection layers. Defaults to `False`.

        Raises:
            ValueError:
                 If `dims` is not divisible by `num_heads`.


        """
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        """
        Performs a single call of the multi-head attention mechanism on provided queries, keys, and values.
        This function processes the input tensors by first projecting them into query, key, and value representations using respective fully connected layers. It then splits the output along the head dimension to facilitate multi-head attention. The scaled dot-product attention function is then applied, optionally taking a mask argument to prevent certain positions from attending to others. Finally, the attended outputs are concatenated and projected back to the original dimensionality of the inputs.

        Args:
            queries (torch.Tensor):
                 Tensor containing query vectors. Shape is expected to be (batch_size, sequence_length, model_dim).
            keys (torch.Tensor):
                 Tensor containing key vectors. Shape is expected to be (batch_size, respective_sequence_length, model_dim).
            values (torch.Tensor):
                 Tensor containing value vectors. Shape should match that of 'keys'.
            mask (torch.Tensor, optional):
                 Mask tensor to prevent certain positions from being attended to. Defaults to None.

        Returns:
            (torch.Tensor):
                 The output tensor after applying multi-head attention. Shape is (batch_ptr_size, sequence_length, model_dim).

        Raises:
            Any exceptions raised during tensor operations will be propagated up to the caller.

        """
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(output)


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) neural network module extending Torch's `nn.Module`.
    This module represents a basic fully connected neural network (also known as a multilayer perceptron) with one hidden layer. It includes an activation function between the two linear transformations.

    Attributes:
        activation_fn (nn.GELU):
             The Gaussian Error Linear Unit (GELU) activation function used after the first linear
            layer with an option to use an approximation for faster computation.
        fc1 (nn.Linear):
             The first linear transformation layer, which maps from input feature space to the intermediate space.
        fc2 (nn.Linear):
             The second linear transformation layer, which maps from the intermediate space back to the original feature space.
            The structure is defined during initialization where the configuration and dimensions of the linear layers are provided through a `VisionConfig` object.

    Args:
        config (VisionConfig):
             A configuration object containing parameters for the MLP such as the size of the hidden
            layer (intermediate_size) and size of the input/output (hidden_size).
            The forward pass of data through the model is executed by calling the instance with input data `x`. The input is transformed by the first linear layer, passed through the GELU activation function, and then transformed by the second linear layer to produce the output.

    Args:
        x (mx.array):
             The input data to the MLP model.

    Returns:
        (mx.array):
             The output of the MLP model after processing the input data `x` through two linear layers and a non-linear activation function.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of the neural network module with specific configurations.

        Args:
            config (VisionConfig):
                 An instance of the VisionConfig class that contains
                configuration parameters such as hidden_size and intermediate_size.

        Raises:
            ValueError:
                 If the `activation_fn` is not recognized as a valid activation function.

        """
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Applies a two-layer transformation to the input data.
        This method performs a sequential operation on the input data `x`. First, it applies a linear transformation
        followed by an activation function as defined in `self.fc1` and `self.activation_fn`. Then,
        it applies another linear transformation using `self.fc2`. The transformations correspond to
        layers in a neural network and are intended to be learned parameters that map the input data to a desired
        output format, typically in the context of neural network inference.

        Args:
            x (mx.array):
                 A tensor representing the input data.

        Returns:
            (mxarray):
                 The tensor after applying the two linear transformations and the activation function.

        Raises:
            The method itself does not raise any exceptions, but if the input types and shapes do not align with what
            the layers expect, or if the underlying operations raise errors, those will be propagated to the caller.

        """
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    EncoderLayer is a subclass of `nn.Module` that encapsulates a single layer of a transformer-style encoder.
    Each EncoderLayer performs a sequence of operations on an input tensor `x`. It first normalizes the input using
    layer normalization, then applies a multi-head self-attention mechanism, followed by another layer normalization
    and a multi-layer perceptron (MLP). The self-attention and MLP operations include residual connections, ensuring
    the flow of gradients during backpropagation.

    Attributes:
        embed_dim (int):
             The size of the embedding dimension, derived from the hidden size configuration.
        self_attn (Attention):
             The Attention module that performs multi-head self-attention.
        layer_norm1 (nn.LayerNorm):
             The first layer normalization module.
        mlp (MLP):
             The multi-layer perceptron module following the self-attention.
        layer_norm2 (nn.LayerNorm):
             The second layer normalization module post MLP.

    Args:
        config (VisionConfig):
             Configuration object containing various parameters like hidden size, number
            of attention heads, and layer normalization epsilon value.
        Call Arguments:
        x (mx.array):
             A tensor containing the input features to the EncoderLayer.
        mask (Optional[mx.array]):
             An optional tensor containing a mask to be applied during attention.
            If `None`, no mask is applied.

    Returns:
        (mx.array):
             The output tensor from the EncoderLayer after performing the sequence of operations
            on the input tensor `x`.

    Raises:
        NotImplementedError:
             If any operation invoked within the layer is not implemented.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of a network layer with embedded attention and MLP components.

        Args:
            config (VisionConfig):
                 An instance of VisionConfig class which provides necessary parameters such as hidden_size, num_attention_heads, and layer_norm_eps to configure the layer.

        Raises:
            ValueError:
                 If the `hidden_size` is not divisible by `num_attention_heads` in the Attention model.

        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Performs a forward pass on the Transformer block with optional masking.
        This method applies the Transformer block's operations on the input data `x`. It first normalizes `x` using a layer normalization, then computes self-attention with an optional mask, adds the self-attention results to the original input (residual connection), and then passes this through another layer normalization followed by a multi-layer perceptron (MLP) block. The output of the MLP block is added to the input after the first residual connection to form the final output.

        Args:
            x (mx.array):
                 The input data to be processed by the Transformer block.
            mask (Optional[mx.array]):
                 An optional mask to be applied during the self-attention operation. If provided, it determines which positions should be attended to. Defaults to None.

        Returns:
            (mx.array):
                 The output of the Transformer block after applying layer normalization, self-attention, residual connections, and MLP operations on the input `x`.

        """
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """
    A neural network module that represents an encoder architecture.
    The Encoder class is a component in a vision-based model and is responsible for transforming inputs through a series of encoder layers, as defined by a given configuration.

    Attributes:
        layers (List[EncoderLayer]):
             A list of encoder layers that will process the inputs. The number and configuration of these layers are determined by the `config` parameter, specifically the attribute `num_hidden_layers`.

    Args:
        config (VisionConfig):
             An instance of the VisionConfig class which contains configuration parameters such as the number of hidden layers and other encoder-specific settings.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance with a set of EncoderLayers.

        Args:
            config (VisionConfig):
                 A configuration object containing parameters for building the encoder layers.
                The __init__ method constructs the layers attribute by instantiating multiple EncoderLayer
                objects according to the 'num_hidden_layers' specified in the VisionConfig.

        """
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class VisionEmbeddings(nn.Module):
    """
    A nn.Module subclass that computes embeddings for vision tasks.
    This module is designed to convert image tensors into a sequence of embeddings that can be processed by
    transformer models. It breaks down images into patches, embeds them individually, and concatenates a
    learnable class embedding to the sequence of patch embeddings. Positional embeddings are also added to
    the sequence to retain positional information of the patches.

    Attributes:
        config (VisionConfig):
             Configuration object containing model hyperparameters.
        embed_dim (int):
             Dimensionality of the embeddings.
        image_size (int):
             Size of the input images.
        patch_size (int):
             Size of the patches the image is divided into.
        class_embedding (mx.ndarray):
             A learnable embedding for the class token.
        patch_embedding (nn.Conv2d):
             Convolutional layer to create patch embeddings from images.
        num_patches (int):
             Total number of patches to be extracted from the image.
        num_positions (int):
             Total positions for which positional embeddings are created, including the class position.
        position_embedding (nn.Embedding):
             Embedding layer for encoding position information.
        The module takes an image tensor and processes it through the following steps:
            1. Use the `patch_embedding` convolutional layer to extract embeddings for each patch of the image.
            2. Concatenate a class embedding to the sequence of patch embeddings.
            3. Create a sequence of position IDs corresponding to each patch and the class embedding.
            4. Add positional embeddings to the patch and class embeddings.
            The result is a sequence of embeddings that can be fed into subsequent layers of a transformer model.

    Args:
        config (VisionConfig):
             A configuration object that includes settings like image size, patch size, and embedding dimension.
        Call Arguments:
        x (mx.ndarray):
             A batch of images to be processed.

    Returns:
        (mx.ndarray):
             The sequence of embeddings for the input images, including positional and class embeddings.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new vision model with the given configuration.

        Args:
            config (VisionConfig):
                 An instance of the VisionConfig class that
                provides configuration parameters for the vision model.

        Raises:
            ValueError:
                 If any of the configuration parameters are invalid.
            The method initializes several attributes for the vision model:
            - embed_dim:
                 The dimension of the embedding space.
            - image_size:
                 The size to which input images will be resized.
            - patch_size:
                 The size of the patches that the image will be divided into.
            - class_embedding:
                 A zeros-initialized embedding vector for class tokens.
            - patch_embedding:
                 A convolutional layer that prepares the patch embeddings.
            - num_patches:
                 The number of patches based on the provided image and patch sizes.
            - num_positions:
                 The total number of positions which includes all patches and one class token.
            - position_embedding:
                 An embedding layer that captures positional information for patches and the class token.
                The '__init__' method also inherits and invokes the initialization of its superclass.

        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.zeros((config.hidden_size,))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Applies the object's embedding logic to input data to generate a transformed representation.
        This method processes the input array `x` by first generating patch embeddings through the object's 'patch_embedding' method. It then flattens these embeddings appropriately. A class token embedding is broadcasted to match the batch size and concatenated with the patch embeddings. To consider positional information, position embeddings are added to this concatenated result. The final output is a representation that includes class and positional information to be used downstream.

        Args:
            x (mx.array):
                 A batch of input images, where the first dimension represents the batch size.

        Returns:
            (mx.array):
                 The resulting array after applying the class embedding, patch embedding, positional embedding, and any additional transformations defined in the '__call__' method.

        """
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, embed_dim)
        )
        position_ids = mx.array(np.arange(self.num_positions)[None, :])

        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding(position_ids)
        return embeddings


class ClipVisionModel(nn.Module):
    """
    A PyTorch module representing the vision part of a CLIP (Contrastive Language-Image Pretraining) model.
    This class implements a vision model structure inspired by the architecture used in CLIP models, which are designed to
    understand images in the context of natural language. It combines embedding layers, layer normalization and an
    encoder to process visual input.

    Attributes:
        embeddings (VisionEmbeddings):
             An instance of VisionEmbeddings which produces initial embeddings from input images.
        pre_layrnorm (nn.LayerNorm):
             Layer normalization applied before the encoder layers.
        encoder (Encoder):
             The Encoder instance containing the layers responsible for the bulk of processing in the vision model.
        post_layernorm (nn.LayerNorm):
             Layer normalization applied after the encoder layers to the pooled output.

    Args:
        config (VisionConfig):
             A configuration object containing model dimensions and specifications specific to the
            vision part of the CLIP model.
        The module's forward method accepts two arguments:
        x (mx.array):
             A batch of image tensors that will be processed by the model.
        output_hidden_states (Optional[bool]):
             A flag indicating whether the model should output the hidden states of
            each layer. By default, no hidden states are returned (if None is provided).

    Returns:
        (mx.array):
             A tuple containing the following elements:
        (- pooler_output):
             The pooled output features from the model which can be used for downstream tasks.
        (- x):
             The final layer output features from the model.
        (- encoder_states):
             The hidden states at each layer of the model, returned only if output_hidden_states is True.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of a neural network model which includes the necessary components for processing and encoding visual data.

        Args:
            config (VisionConfig):
                 A configuration object containing parameters for the neural network, including the number of hidden layers, the hidden layer size, image dimensions, and other hyperparameters relevant to the model architecture.

        Raises:
            ValueError:
                 If any of the configuration values provided in 'config' are invalid or not compatible with the expected model architecture.

        """
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        """
        Processes input data through the vision model's embedding, normalization, and encoding layers, with optional output of hidden states.
        This method is the callable entry point for passing input through the vision model. It begins by applying embeddings to the input data, followed by a layer normalization step. It then sequentially processes the data through each encoder layer, optionally collecting the intermediate hidden states. Finally, it applies a post-layer normalization to the output of the first token (assumed to be a special token for pooling) to produce the 'pooler output'. This step is usually used to extract a fixed size representation from variable length inputs which is useful for tasks like classification.

        Args:
            x (mx.array):
                 The input data tensor to the model.
            output_hidden_states (Optional[bool]):
                 Flag to decide whether to return all hidden states.

        Returns:
            (mx.array):
                 A tuple containing the pooler output, the final hidden state, and optionally the encoder hidden states if `output_hidden_states` is True.

        """
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)

        encoder_states = (x,) if output_hidden_states else None

        for l in self.encoder.layers:
            x = l(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        pooler_output = self.post_layernorm(x[:, 0, :])
        return pooler_output, x, encoder_states


class VisionModel(nn.Module):
    """
    A PyTorch module representing a vision model, specifically configured for CLIP-based vision tasks.
    The VisionModel class is designed to encapsulate a vision model that is compatible with the
    CLIP (Contrastive Language–Image Pretraining) model's vision component. It initializes with a given
    configuration and checks for compatibility with the 'clip_vision_model' type. The class provides methods
    for processing inputs through the encapsulated CLIP-based vision model and sanitizing model weights
    during state dict loading or model loading routines.

    Attributes:
        model_type:
             A string representing the type of model, extracted from the configuration. It must be
            'clip_vision_model' to be supported.
        vision_model:
             An instance of ClipVisionModel that performs vision-related processing.

    Methods:
        __init__(self, config:
             VisionConfig):
            Initializes the VisionModel class.

    Args:
        config:
             A VisionConfig object containing the configuration for the vision model.

    Raises:
        ValueError:
             If the model_type specified in the config is not 'clip_vision_model'.
        __call__(self, x:
             mx.array, output_hidden_states: Optional[bool]=None) -> mx.array:
            Processes the input data through the vision model and returns the output.

    Args:
        x:
             The input array to the vision model.
        output_hidden_states:
             Optional; A boolean indicating whether to output hidden states.

    Returns:
        The result of the vision model processing the input data.
        sanitize(self, weights):
            Sanitizes the weights dictionary, ensuring compatibility and consistency before loading.

    Args:
        weights:
             A dictionary of weights representing the state of the model.

    Returns:
        A sanitized dictionary of weights with correct shapes and keys.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of a CLIP vision model.
        This constructor method for the vision model requires a configuration object of type VisionConfig. Upon instantiation, it initializes the model with specified configurations. It raises a ValueError if an unsupported model type is passed to the configuration.

        Args:
            config (VisionConfig):
                 The configuration object specifying the model parameters.

        Raises:
            ValueError:
                 If the `model_type` attribute in the config object is not set to 'clip_vision_model', a ValueError exception is raised indicating an unsupported model type.

        """
        super().__init__()

        self.model_type = config.model_type
        if self.model_type != "clip_vision_model":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = ClipVisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        """
        Performs a forward pass on the vision model with the given input.

        Args:
            x (mx.array):
                 The input data to the vision model.
            output_hidden_states (Optional[bool], optional):
                 A flag to decide whether to output hidden states or not. Default is None, where
                the default behavior of the model will be used.

        Returns:
            (mx.array):
                 The output of the vision model for the given input.

        """
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights):
        """
        Sanitizes a dictionary of weights by removing certain keys and potentially transposing others.
        This function iterates through the key-value pairs in the input dictionary `weights`. If a key
        contains 'position_ids', it is skipped and not included in the sanitized dictionary. If a key
        contains 'patch_embedding.weight' and the corresponding value (an array representing weights)
        passes a specific shape check via `check_array_path()`, it is included as-is; otherwise, it is
        included after being transposed to adjust the shape. All other key-value pairs are included
        without modification. The purpose of this sanitation process is to prepare weights in a
        format expected by a specific model or system without certain elements and with others
        shaped correctly.

        Args:
            weights (dict):
                 A dictionary where keys are strings representing the name of the weight
                variables, and values are the weight arrays to be sanitized.

        Returns:
            (dict):
                 A new dictionary with sanitized weights.

        """
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
