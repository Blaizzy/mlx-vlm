"""

The `vision` module contains implementations for various components involved in processing visual inputs using a neural network model, specifically designed for image analysis tasks. The module includes numerous classes and functions encapsulating different aspects of a vision model architecture, such as configurations, layers, and blocks that form a typical vision transformer (ViT) or similar architecture.

The module comprises the following primary classes and functions:

- `VisionConfig`: A data class that holds configuration parameters for defining a vision model's structure, such as model type, number of layers, sizes of hidden layers, and other model hyperparameters. The class also includes a method `from_dict` to instantiate a `VisionConfig` object from a dictionary of parameters, ensuring that only valid parameters specified in the class signature are used.

- `check_array_shape`: A utility function to verify if a given NumPy array's shape adheres to specific criteria. Typically, this function checks that tensor shapes are appropriate for kernel weights of convolutional layers.

- `Attention`, `MHA` (Multi-Head Attention), and `MLP` (Multi-Layer Perceptron) classes: These classes implement key components of a transformer model, each serving a specific purpose in the model. The `Attention` class provides scaled dot-product attention mechanism, while `MHA` represents a multi-head attention block, and `MLP` serves as a feedforward neural network layer.

- `EncoderLayer` and `Encoder`: These classes model a single layer and a stack of layers of a transformer's encoder, respectively. They incorporate normalizations, attention mechanisms, and feedforward networks to transform input features.

- `VisionEmbeddings`: This class is responsible for processing raw image inputs into a form suitable for further processing by the encoder. It includes a patch embedding convolution to slice images into patches and positionally encode them.

- `SigLipVisionModel`: A vision model class that assembles the components mentioned earlier into a cohesive model structure, including embeddings, encoder, and additional layer normalization and pooling heads.

- `SigLipMultiheadAttentionPoolingHead`: A class that represents a specialized pooling head with multi-head attention, used to aggregate features from the encoder into a pooled representation.

- `VisionModel`: A high-level class that incorporates the `SigLipVisionModel`. It serves as the main entry point for processing images through the neural network, and it offers an interface for performing forward passes and handling model weights.

Additionally, the `sanitize` method within the `VisionModel` class ensures that weights conform to expected tensor shapes before applying them to the model.

Overall, the `vision` module provides a comprehensive set of tools for building and handling advanced vision models, with a particular focus on attention-driven architectures suitable for a variety of image-based machine learning tasks.
"""

import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    """
    A dataclass to represent the configuration for a vision-based model.

    Attributes:
        model_type (str):
             The type of the model to configure.
        num_hidden_layers (int):
             The number of hidden layers in the model (default: 27).
        hidden_size (int):
             The size of each hidden layer (default: 1152).
        intermediate_size (int):
             The size of the intermediate layer (default: 4304).
        num_attention_heads (int):
             The number of attention heads in the model (default: 16).
        image_size (int):
             The input image size in pixels (default: 384).
        patch_size (int):
             The size of the patches the image is divided into (default: 14).
        projection_dim (int):
             The dimension of the output tokens or embeddings (default: 768).
        vocab_size (int):
             The size of the vocabulary or the number of classes (default: 32000).
        num_channels (int):
             The number of channels in the input images (default: 3).
        layer_norm_eps (float):
             The epsilon value for the layer normalization (default: 1e-06).
        Class Methods:
        from_dict:
             A class method that creates an instance of VisionConfig from a dictionary of parameters. This method inspects the class signature to ensure only valid attributes are extracted and used to instantiate the class.

    """

    model_type: str
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, params):
        """
        Constructs an instance of a class from a dictionary containing parameter names and values.
        This class method iterates over the key/value pairs in the given dictionary, filtering out any pairs where the key does not match a parameter name in the class constructor. It then uses the remaining pairs to instantiate the class using keyword argument unpacking.

        Args:
            params (dict):
                 A dictionary with keys corresponding to the class constructor's parameter names and values being the values to be used for instantiating the class. Keys that are not parameter names of the class constructor will be ignored.

        Returns:
            An instance of the class, constructed with the provided parameters from the dictionary.

        Raises:
            TypeError:
                 If any key in the dictionary does not correspond to a parameter name in the class constructor, or if a value is provided that is not compatible with the type expected by the constructor.

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
    Checks if the provided array has a specific 4-dimensional shape with constraints.
    This function verifies whether an array has four dimensions and if the number
    of output channels is greater than or equal to the height and
    width of the kernel which must be equal to each other.

    Args:
        arr (numpy.ndarray):
             The array to be checked for the specific shape criteria.

    Returns:
        (bool):
             True if the array has the required shape characteristics, False otherwise.

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
    A class that implements a multi-head attention mechanism as a module.

    Attributes:
        num_heads (int):
             The number of attention heads.
        scale (float):
             Normalization factor for the dot products.
        q_proj (nn.Linear):
             Linear projection layer for query.
        k_proj (nn.Linear):
             Linear projection layer for key.
        v_proj (nn.Linear):
             Linear projection layer for value.
        out_proj (nn.Linear):
             Linear projection layer to produce final output.

    Args:
        dims (int):
             The dimensionality of the query and key vectors.
        num_heads (int):
             The number of parallel attention heads.
        query_input_dims (Optional[int]):
             The dimensionality of the input query vectors. Defaults to None, which means it's set to `dims`.
        key_input_dims (Optional[int]):
             The dimensionality of the input key vectors. Defaults to None, which means it's set to `dims`.
        value_input_dims (Optional[int]):
             The dimensionality of the input value vectors. Defaults to None, which is then set to the value of `key_input_dims`.
        value_dims (Optional[int]):
             The dimensionality to project the values to. Defaults to None, which means it's set to `dims`.
        value_output_dims (Optional[int]):
             The dimensionality of the final output values after combining with the attention weights. Defaults to None, which means it's set to `dims`.
        bias (bool):
             If True, adds a learnable bias to the projection layers. Defaults to False.

    Raises:
        ValueError:
             If `dims` is not divisible by `num_heads`.

    Methods:
        __call__:
             Defines the computation performed at every call of the Attention module. Takes in queries, keys, values, and an optional mask, performs the attention mechanism, and returns the result of scaled dot-product attention combined with an output projection.

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
        Initializes a multi-head attention module with optional specifications for different input dimensions and projections.

        Args:
            dims (int):
                 The number of expected features in the input.
            num_heads (int):
                 The number of heads in the multi-head attention mechanism.
            query_input_dims (Optional[int]):
                 The dimension of the query input space. Defaults to 'dims' if not provided.
            key_input_dims (Optional[int]):
                 The dimension of the key input space. If not provided, defaults to the same value as 'query_input_dims'.
            value_input_dims (Optional[int]):
                 The dimension of the value input space. If not provided, defaults to 'key_input_dims'.
            value_dims (Optional[int]):
                 The dimension of the value projection. If not provided, defaults to 'dims'.
            value_output_dims (Optional[int]):
                 The dimension of the output space after attention weights have been applied. Defaults to 'dims' if not provided.
            bias (bool):
                 If set to True, layers will learn an additive bias. Defaults to False.

        Raises:
            ValueError:
                 If 'dims' is not divisible by 'num_heads', an exception is raised as this is a requirement for equal division of dimensions across heads.

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

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        """
        Performs the forward pass of the multi-head attention mechanism.
        This method computes scaled dot product attention over the provided
        queries, keys, and values, with an optional mask. The queries, keys, and values
        are first linearly projected, and then reshaped to fit the multi-head
        attention pattern. The attention mechanism allows the model to focus
        on different parts of the input sequence when making predictions.
        The output of the attention operation is then projected back to the
        expected dimensionality for further processing.

        Args:
            queries (Tensor):
                 The query set with shape (batch_size, query_len, embed_dim).
            keys (Tensor):
                 The key set with shape (batch_name, key_len, embed_dim).
            values (Tensor):
                 The value set with shape (batch_name, key_len, embed_dim).
            mask (Optional[Tensor]):
                 An optional binary mask with shape (batch_name, query_len, key_len)
                that can be used to prevent attention to certain positions.

        Returns:
            (Tensor):
                 A tensor containing the computed attention values with shape
                (batch_size, query_len, embed_dim).

        Raises:
            ValueError:
                 If the input dimensions are invalid or the mask shape is incompatible.

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


class MHA(nn.Module):
    """
    A class to implement the Multi-Head Attention (MHA) mechanism as a module.
    This class defines a Multi-Head Attention mechanism as described in 'Attention
    Is All You Need' paper. The MHA module is implemented as a PyTorch module and can be
    integrated into neural network architectures for various tasks that require
    self-attention mechanisms.

    Attributes:
        num_heads (int):
             The number of heads in the multi-head attention mechanism.
            Each head computes an independent attention response before they are combined.
        scale (float):
             The scaling factor for query normalization, computed as head_dim^(-0.5).
        in_proj (nn.Linear):
             A linear layer that projects the input queries into a
            concatenated sequence of queries, keys, and values.
        out_proj (nn.Linear):
             A linear layer that projects the computed attention
            responses back to the required dimensions.

    Raises:
        ValueError:
             An error occurs if `dims` is not evenly divisible by `num_heads`.

    Args:
        dims (int):
             The dimensionality of the input features.
        num_heads (int):
             The number of attention heads to use.
        bias (bool, optional):
             If True, includes bias terms in the projection layers.
            Defaults to False.

    Note:
        The MHAclass does not implement its own forward method; rather, it overloads
        the call method to allow the module to be called directly with its inputs.

    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = False,
    ):
        """
        Initializes a multi-head attention layer.
        This constructor initializes a multi-head attention layer with the specified number of
        dimensions (dims), number of heads (num_heads), and an optional bias parameter. The dims
        parameter represents the input and output dimensionality of the layer, while num_heads
        is the number of heads in the multi-head attention mechanism. The bias parameter is
        a boolean indicating whether the linear transformations should include a bias term.
        The initialization process checks if the provided dimensions are evenly divisible by
        the number of heads, as each head processes a slice of the input feature space. If not,
        a ValueError is raised. Additionally, the layer's scaling factor (self.scale) is
        computed based on the dimensions of each head, which is used to scale attention scores.
        The in_proj linear layer combines the linear transformations for the queries, keys, and
        values into a single weight matrix, and out_proj is the final linear transformation
        applied after the attention scores have been computed and applied.

        Parameters:
            dims (int):
                 The dimensionality of input and output features.
            num_heads (int):
                 The number of attention heads.
            bias (bool):
                 A flag determining whether linear transformation layers should have a bias term; defaults to False.

        Raises:
            ValueError:
                 If the input feature dimensions are not divisible by the number of heads.

        """
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.in_proj = nn.Linear(dims, dims * 3, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries: mx.array, kv: mx.array, mask=None, cache=None):
        """
        Performs the forward pass through the multi-head attention mechanism.
        Takes a batch of query matrices, a batch of key-value pairs, and an optional mask to control the attention's focus. Projects the queries, keys, and values using a learned linear transformation, applies scaled dot-product attention, then projects the result to the appropriate output dimension.

        Args:
            queries (mx.array):
                 The query matrices with shape (B, L, D) where B is the batch size, L is the sequence length, and D is the dimension of each query.
            kv (mx.array):
                 The key and value matrices. It is assumed that keys and values are combined in the kv array in the shape (B, S, 2*D), where S is the sequence length for keys and values.
            mask (Optional):
                 An optional mask to apply on the attention weights. Can be used to hide certain elements from the model's view within each attention head.
            cache (Optional):
                 An optional cache where intermediate results can be stored for efficiency.

        Returns:
            (mx.array):
                 An output array with the shape (B, L, -1) where B is the batch size and L is the sequence length. It represents the transformed query matrices.

        Raises:
            ValueError:
                 If any of the inputs are not as expected or if the shapes of the inputs do not align as required by the multi-head attention mechanism.

        """
        B, L, D = queries.shape

        qkv = self.in_proj(queries)
        _, keys, values = mx.split(qkv, 3, axis=-1)

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
    A Multi-Layer Perceptron (MLP) module representing a fully connected feedforward neural network.
    This class inherits from `nn.Module` and is initialized with the network architecture defined by the `VisionConfig` object. It contains two fully connected layers with an activation function in between.

    Attributes:
        activation_fn (nn.Module):
             An activation function layer. Here, GeLU (Gaussian Error Linear Unit) is used with a fast approximation.
        fc1 (nn.Linear):
             The first fully connected layer that transforms input features to an intermediate representation.
        fc2 (nn.Linear):
             The second fully connected layer that transforms the intermediate representation back to the original feature space.

    Args:
        config (VisionConfig):
             A configuration object containing parameters such as hidden and intermediate layer sizes.

    Methods:
        __call__(self, x:
             mx.array) -> mx.array:
            Defines the computation performed at every call of the MLP module. It applies the first linear transformation,
            follows it with the activation function, and then applies the second linear transformation.

    Args:
        x (mx.array):
             The input feature array to the MLP.

    Returns:
        (mx.array):
             The output of the MLP after processing the input feature array.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of the class, setting up the neural network layers based on the provided configuration settings.

        Args:
            config (VisionConfig):
                 A configuration object containing parameters such as hidden sizes and the intermediate size.
                The method sets up the activation function as GELU (with a fast approximation) and initializes two fully connected (linear) layers. The first linear layer maps from the hidden size specified in the config to the intermediate size, and the second linear layer maps back from the intermediate size to the hidden size.

        """
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Performs forward propagation through the network layers.
        This method represents the forward propagation algorithm for a neural network.
        It takes an input tensor `x`, applies the first fully connected layer (`fc1`), then
        passes the result through an activation function, followed by a second fully
        connected layer (`fc2`). The method returns the output of the network.

        Args:
            x (mx.array):
                 A multidimensional array of input data on which the forward
                propagation must be performed.

        Returns:
            (mx.array):
                 The result of the network after applying the two fully connected
                layers and the activation function on the input data `x`.

        """
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    A single layer for an encoder structure that applies self-attention and a feedforward neural network (MLP).
    This class inherits from `nn.Module` and represents a single layer of an encoder architecture typically used in vision models. It applies a self-attention mechanism followed by a multi-layer perceptron (MLP) to its input. Layer normalization is applied before both the self-attention and MLP to stabilize learning.

    Attributes:
        embed_dim (int):
             The dimensionality of the input embeddings.
        self_attn (Attention):
             The self-attention module used within the layer.
        layer_norm1 (nn.LayerNorm):
             The first layer normalization applied prior to the self-attention mechanism.
        mlp (MLP):
             The multi-layer perceptron module applied after the self-attention mechanism.
        layer_norm2 (nn.LayerNorm):
             The second layer normalization applied prior to the MLP module.

    Args:
        config (VisionConfig):
             A configuration object containing parameters for the layer’s submodules, including the hidden size, number of attention heads, and layer normalization epsilon value.
        Callable with:
        x (mx.array):
             The input tensor to be processed by the encoder layer.
        mask (Optional[mx.array], default=None):
             An optional mask to be applied to the self-attention mechanism.

    Returns:
        (mx.array):
             The output tensor after applying the self-attention and MLP to the input tensor, with residual connections and layer normalization applied.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the transformer block with specified configurations.

        Args:
            config (VisionConfig):
                 A class containing the configuration settings for the transformer block.
            It should have the following attributes:
            - hidden_size:
                 An integer representing the dimensionality of the embeddings.
            - num_attention_heads:
                 An integer representing the number of attention heads to use
                within the multi-head attention mechanism.
            - layer_norm_eps:
                 A floating-point number representing the epsilon value to use
                for layer normalization layers to prevent divide by zero errors.

        Raises:
            TypeError:
                 If the `config` argument is not of type `VisionConfig`.

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
        Performs a forward pass on a Transformer block.

        Args:
            x (mx.array):
                 The input tensor to the Transformer block.
            mask (Optional[mx.array]):
                 An optional mask tensor to be applied to the self-attention layer, with default of None.

        Returns:
            (mx.array):
                 The output tensor after applying the Transformer block operations.
            (This function sequentially applies the following operations to the input tensor):
                1. Layer normalization (first layer).
                2. Multi-head self-attention with optional masking.
                3. Residual connection by adding the output of the multi-head self-attention to the input tensor.
                4. Layer normalization (second layer).
                5. Feed-forward neural network (MLP).
                6. Residual connection by adding the output of the MLP to the result of the second layer normalization.

        """
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """
    A neural network module designed for encoding visual input.
    The Encoder class is a subclass of `nn.Module` and is responsible for creating a series of encoder layers
    that are stacked on top of each other. This structure is typical in transformer architectures where
    sequential layers each perform operations on their input data and pass the results onward. The Encoder
    is initialized based on a configuration object that specifies parameters such as the number of hidden layers.

    Attributes:
        layers (List[EncoderLayer]):
             A list of `EncoderLayer` instances which constitute the encoder part of the network.
            Each `EncoderLayer` is created with the configuration provided by the `config` argument.

    Args:
        config (VisionConfig):
             An instance of `VisionConfig` containing configuration parameters like the number of
            hidden layers to be used in creating the Encoder layers.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of the network with the given configuration.

        Args:
            config (VisionConfig):
                 A configuration object that contains settings
                for the model architecture, such as the number of hidden layers,
                hidden size, number of attention heads, and layer normalization
                epsilon.
                Each encoder layer in the network is created based on the provided configuration.
                The encoder layers make up the core of the network, allowing for intricate
                representations to be learned. The initialized layers are stored as a list
                within the object's 'layers' attribute.

        """
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class VisionEmbeddings(nn.Module):
    """
    A PyTorch nn.Module for extracting embeddings from visual inputs using patch embedding and position embedding techniques.
    This module is designed to process images by dividing them into smaller patches and encoding the
    patches into embeddings, which are then combined with positional information to preserve the spatial
    arrangement. This is a common approach in vision transformer architectures.

    Attributes:
        config (VisionConfig):
             Configuration object containing various parameters for the model.
        embed_dim (int):
             The dimensionality of the embeddings.
        image_size (int):
             The size (height/width) of the input images.
        patch_size (int):
             The size of each patch the image is divided into.
        patch_embedding (nn.Conv2d):
             The convolutional layer used for extracting patch embeddings.
        num_patches (int):
             The total number of patches obtained from an image.
        num_positions (int):
             The total number of positional embeddings corresponding to the patches.
        position_embedding (nn.Embedding):
             A learnable embedding layer for positional encoding of patches.
            The `__init__` method initializes the module's layers and internal state based on the passed `config`.
            The `__call__` method is responsible for transforming a batch of image tensors into their
            corresponding embeddings by applying the patch embedding convolution and adding positional embeddings.

    Args:
        self :
             The VisionEmbeddings instance.
        x (mx.array):
             A batch of image tensors expected to be in (batch_size, channels, height, width) format.

    Returns:
        (mx.array):
             The resulting batch of embeddings, with each embedding reflecting both the content
            of the corresponding image patch and its position within the image.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of a Vision model with the provided configuration.

        Args:
            config (VisionConfig):
                 An object containing the configuration settings for the model.

        Raises:
            ValueError:
                 If either the image_size or patch_size is not divisible evenly, resulting in a non-integer number of patches.

        Attributes:
            config (VisionConfig):
                 The configuration object containing settings such as hidden_size, image_size, etc.
            embed_dim (int):
                 The dimension of the embedding layer, set to the hidden_size from the config.
            image_size (int):
                 The size of the input images, as specified by the config.
            patch_size (int):
                 The size of each patch the image is divided into, according to the config.
            patch_embedding (torch.nn.Conv2d):
                 A 2D convolutional layer that creates patch embeddings.
            num_patches (int):
                 The total number of patches derived from the image_size and patch_size.
            num_positions (int):
                 The number of positional embeddings, equivalent to num_patches.
            position_embedding (torch.nn.Embedding):
                 A layer for learning positional embeddings.

        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Transforms the input data into position-aware embeddings.
        This method applies patch embedding to the input matrix, flattens the result,
        and adds positional embeddings. The method is used to prepare the input data
        before passing it through subsequent layers of a neural network model.

        Args:
            x (mx.array):
                 The input matrix with a shape that includes the batch size as its first dimension.

        Returns:
            (mx.array):
                 The resulting embedded and position-aware flattened matrix suitable for further processing by the model.

        Raises:
            ValueError:
                 If the input matrix x does not contain the batch size as its first dimension, this method will raise an error due to mismatched dimensions during the patch embedding or flattening process.

        """
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        self.position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings
        embeddings += self.position_embedding(self.position_ids)
        return embeddings


class SigLipVisionModel(nn.Module):
    """
    A neural network module for vision tasks, specifically designed to integrate signature and lip attributes using a transformer-like architecture.
    This class inherits from `torch.nn.Module` and represents the core model which is composed of embedding layers, an encoder, a post-layer normalization layer, and a specialized pooling head for attention mechanisms. It encapsulates the entire process from the initial input to the output stage, combining these components into a unified architecture for processing visual inputs.

    Attributes:
        embeddings (VisionEmbeddings):
             An instance of VisionEmbeddings, handling the initial transformation of input data.
        encoder (Encoder):
             An instance of Encoder, consisting of multiple layers that encode the input embeddings into a sequence of representations.
        post_layernorm (torch.nn.LayerNorm):
             A layer normalization module applied after the encoding process.
        head (SigLipMultiheadAttentionPoolingHead):
             The pooling head implementing a multi-head attention mechanism, designed for the specific task of signature and lip vision tasks.

    Args:
        config (VisionConfig):
             A configuration object containing model hyperparameters and settings.

    Methods:
        __call__(x:
             mx.array, output_hidden_states: Optional[bool]=None) -> mx.array: Processes the input through the embeddings, encoder, and head, and returns a tuple of the pooled output, last hidden state, and optionally the encoder states.

    Args:
        x (mx.array):
             The input batch of images.
        output_hidden_states (Optional[bool]):
             A flag indicating whether to return the hidden states of the encoder layers. Defaults to None.

    Returns:
        (mx.array):
             A tuple containing the pooled output from the head, the last hidden state, and optionally the hidden states at each layer of the encoder if `output_hidden_states` is set to True.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the neural network component with vision capabilities.
        This method sets up the essential parts of a vision-based neural network model by initializing the embedding layers, encoder layers, post-layer normalization, and the pooling head with an attention mechanism. Each of these components is configured using the provided `VisionConfig` object that defines specific parameters such as hidden sizes, patch sizes, and the number of attention heads.

        Args:
            config (VisionConfig):
                 A configuration object that contains various hyperparameters and settings like number of hidden layers, hidden sizes, and so on, used to tailor the neural network to specific requirements.

        Raises:
            TypeError:
                 If `config` is not an instance of `VisionConfig`.

        """
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        """
        Calls the SigLipVisionModel with an input tensor and optionally returns hidden states.

        Args:
            x (mx.array):
                 The input tensor to the vision model. It should represent the image data
                structured in a way that matches the expected input format of the model. The precise
                shape and preparation of the input tensor depends on the specific vision
                model configuration.
            output_hidden_states (Optional[bool], optional):
                 Determines whether to return the hidden states
                generated during the encoding process. If `True`, the output includes a tuple
                containing the hidden states at each layer of the encoder in addition to the
                standard model outputs. Defaults to None, in which case the hidden states
                are not returned.

        Returns:
            (mx.array):
                 A tuple containing the following elements:
            - pooler_output (mx.array):
                 The output of the pooling head, typically used for representing
                the entire image.
            - x (mx.array):
                 The final encoded state of the input tensor after processing through the vision model.
            - encoder_states (Tuple[mx.array], optional):
                 The hidden states at each encoder layer, returned
                only when `output_hidden_states` is `True`. If `output_hidden_states` is `False`, this
                value is `None`. The hidden states are useful for in-depth model analysis or for
                creating complex architectures atop the base vision model.

        """
        x = self.embeddings(x)

        encoder_states = (x,) if output_hidden_states else None

        for l in self.encoder.layers:
            x = l(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        pooler_output = self.post_layernorm(x[:, 0, :])
        pooler_output = self.head(pooler_output)
        return pooler_output, x, encoder_states


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """
    A neural network module that implements a pooling head with multihead attention for 'vision-like' data structures.
    The module combines a multihead attention mechanism with a learnable probe vector to pool contextual information from a sequence. It also includes layer normalization and a multi-layer perceptron to further process the pooled signal. This module is particularly suitable for tasks that require condensing information from an entire sequence into a single representation, such as in image recognition tasks.

    Attributes:
        probe (mx.ones):
             Learnable parameter vector acting as the query in the attention mechanism. Initialized to a tensor of ones.
        attention (MHA):
             Multihead attention submodule with specified hidden size and number of attention heads.
        layernorm (nn.LayerNorm):
             Layer normalization submodule, stabilizing the activations before passing them through the MLP.
        mlp (MLP):
             Multi-layer perceptron submodule that processes the output from the layer normalization.

    Args:
        config (VisionConfig):
             Configuration object containing the hidden size, number of attention heads, and layer normalization epsilon. This configures the various submodules within the head.

    Methods:
        __call__(self, x:
             mx.array): Forwards the input x through the attention mechanism using the probe as the query, applies a residual connection followed by layer normalization, and then processes the output through the MLP. The method returns a condensed representation of the input at the first position.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the transformer block with attention and feed-forward layers.
        This function initializes a transformer block that is part of a vision model which takes a configuration
        object as input. The function instantiates an `MHA` (Multi-Head Attention) layer, a layer normalization
        layer (`layernorm`), and an `MLP` block (feed-forward layer) according to the specifications provided
        in the `VisionConfig`. It also creates a `probe` vector which is used in certain transformer
        implementations to act as a placeholder for class tokens or similar purposes.

        Args:
            config (VisionConfig):
                 A configuration object which contains various hyperparameters
                such as hidden_size, num_attention_heads, layer_norm_eps, and intermediate_size.

        Raises:
            ValueError:
                 If the hidden_size in `config` is not divisible by num_attention_heads.


        """
        super().__init__()

        self.probe = mx.ones(
            (
                1,
                1,
                config.hidden_size,
            )
        )
        self.attention = MHA(
            config.hidden_size, num_heads=config.num_attention_heads, bias=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array):
        """
        Performs a forward pass on an input tensor using an attention mechanism followed by layer normalization and a feed-forward MLP network.

        Args:
            x (mx.array):
                 The input tensor to which the attention, layer normalization, and MLP layers will be applied.

        Returns:
            (mx.array):
                 The output tensor after applying the attention mechanism to a probing vector, followed by residual connections, layer normalization, and an MLP layer. The output is then sliced to return only the first element along the second dimension.

        Raises:
            None.

        """
        x = self.attention(self.probe, x)[0]

        residual = x
        x = self.layernorm(x)
        x = residual + self.mlp(x)

        return x[:, 0]


class VisionModel(nn.Module):
    """
    A neural network module specifically tailored for vision tasks, adhering to a given configuration.
    VisionModel is a class that extends PyTorch's nn.Module and follows the specified VisionConfig to create a model
    tailored for image processing tasks. It currently supports a specific model type, 'siglip_vision_model', and
    instantiates an internal model of that type upon creation.

    Attributes:
        model_type (str):
             Type of the vision model as specified in the VisionConfig.

    Raises:
        ValueError:
             If the `model_type` specified in the config is not 'siglip_vision_model'.

    Methods:
        __init__(self, config:
             VisionConfig):
            Initializes the VisionModel with the provided VisionConfig. Instantiates the internal vision model using
            the configuration details.
        __call__(self, x:
             mx.array, output_hidden_states: Optional[bool]=None) -> mx.array:
            Allows the VisionModel to be called like a function, passing the input through the internal vision model
            and optionally returning hidden states if specified.
        sanitize(self, weights):
            Validates and adjusts the shapes of the given weights to comply with model requirements, omitting and
            transposing certain layers' weights as necessary for compatibility.

    Note:
        The `sanitize` method is used internally to ensure the model weights are in the correct format for
        the vision tasks. Users of this class are typically not expected to call `sanitize` directly.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the vision model with the given configuration.
        This constructor initializes the vision model with the specified type and configuration.
        If the model type provided through the configuration does not match the expected
        'siglip_vision_model' string, it raises a ValueError indicating that the model type is unsupported.
        Otherwise, it initializes the SigLipVisionModel with the given configuration.

        Args:
            config (VisionConfig):
                 A configuration object containing all necessary information
                to initialize the vision model, including the model type.

        Raises:
            ValueError:
                 If the 'model_type' attribute of the config object is not 'siglip_vision_model'.


        """
        super().__init__()
        self.model_type = config.model_type
        if self.model_type != "siglip_vision_model":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = SigLipVisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        """
        Performs a forward pass through the vision model.

        Args:
            x (mx.array):
                 The input array for the model to process.
            output_hidden_states (Optional[bool], optional):
                 A flag to indicate whether the hidden states should be returned along with the output. Defaults to None.

        Returns:
            (mx.array):
                 The output array after being processed by the model.

        """
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights):
        """
        Sanitizes the input weight dictionary by filtering or altering weights based on specific conditions.
        This method takes a dictionary of weights, iterates through each key-value pair, and processes the weights
        depending on the key. If the key includes 'position_ids', it will be ignored and not included in the output
        sanitized weights. If the key includes 'patch_embedding.weight' and its corresponding value (a multi-dimensional
        array) passes the check_array_shape validation, it will be added to the sanitized weights unchanged. If the
        check_array_shape validation fails, the weight array will be transposed to change the dimension order before
        being added to the sanitized weights. All other key-value pairs are added to the sanitized weights as they are.

        Parameters:
            weights (dict):
                 A dictionary where keys are strings representing the name of the weights, and values are
                the weight arrays to be sanitized.

        Returns:
            (dict):
                 A new dictionary containing the sanitized weights.

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
