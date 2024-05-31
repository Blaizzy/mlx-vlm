"""

The `vision` module provides a collection of classes and functions for constructing and using a neural network tailored for vision tasks. The primary components of the module include the `VisionConfig` data class, which holds configuration parameters for the model; the `EncoderLayer`, `Encoder`, and `VisionEmbeddings` classes that define the individual components of the model architecture; and the `SigLipVisionModel` and the `VisionModel` classes, which serve as the complete vision model structures. Additionally, the module offers utility functions such as `check_array_shape` to assist with input validation, and neural network layers like `Attention` and `MLP` (Multilayer Perceptron) that are integral to the model's operation.

#### Class Definitions and Their Purposes

- `VisionConfig`: A data class that holds various configuration parameters for vision models, such as the number of hidden layers, the size of the hidden layers, and other architectural parameters. Optionally, it can be instantiated from a dictionary of parameters.

- `Attention`: A neural network module that implements the multi-head self-attention mechanism where different portions of the input are paid attention to separately and then combined. It handles projection of queries, keys, and values, and computes the scaled dot-product attention.

- `FastGELUActivation`: This module applies the Gaussian Error Linear Unit (GELU) activation function in a fast approximation, which is used within the MLP to introduce nonlinearity.

- `MLP`: A multilayer perceptron that performs a series of linear transformations with an activation function in between. It is often used after the attention mechanism within transformer models.

- `EncoderLayer`: Represents a single layer of the transformer encoder architecture, including self-attention and feed-forward neural network modules, with layer normalization applied before each module.

- `Encoder`: Constructs a sequence of `EncoderLayer` instances to form the transformer encoder. It allows the option to output hidden states at each layer.

- `VisionEmbeddings`: Responsible for converting input images into a sequence of flattened, positionally-encoded patch embeddings that are suitable for processing by the transformer encoder.

- `SigLipVisionModel`: Assembles the components to form the complete neural network suitable for vision tasks, utilizing the previously described classes. It consists of patch embeddings, a transformer encoder, and layer normalization as the final step.

- `VisionModel`: A wrapper class that instantiates a specific vision model, currently only supporting the 'siglip_vision_model'. It provides a method to call the model as well as a `sanitize` method to clean input weights and ensure compatibility.

#### Utility Functions and Methods

- `check_array_shape`: Checks the shape of an input array to ensure it meets specific criteria, typically used for validating convolutional kernel shapes.

- `from_dict`: A class method provided by `VisionConfig` to create an instance from a dictionary, with parameter checks against the class's constructor signature.

#### Additional Notes

- The `vision` module is primarily designed around a transformer-based architecture for vision tasks, and it employs the conventional components of such models, including attention mechanisms, position embeddings, and multi-layer perceptrons.

- The module appears to be part of a larger machine learning library denoted by the `mlx`, as indicated by the import statements in the code, and could be used as a building block for creating sophisticated image processing models.

- Error handling within the module ensures that the user is informed if they try to instantiate unsupported model types or if configuration parameters are incorrect.

- The implementation is modular, allowing for flexibility and customization when constructing a vision model. Users can extend or modify individual components as needed for specific tasks or datasets.

- The module's design allows for the possibility of outputting intermediate encoder states, thus enabling more advanced use-cases where such information is valuable, like feature extraction or transfer learning.
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
    A dataclass that stores the configuration parameters for a vision model.

    Attributes:
        model_type (str):
             The type of vision model to be configured.
        num_hidden_layers (int):
             The number of hidden layers in the model.
        hidden_size (int):
             The size of each hidden layer.
        intermediate_size (int):
             The size of the intermediate layer in the model.
        num_attention_heads (int):
             The number of attention heads in each attention layer.
        patch_size (int):
             The size of the patches to be extracted from input images for processing.
        projection_dim (int):
             The dimensionality of the projection space for the patches.
        image_size (int):
             The size of the image inputs (defaults to 224).
        num_channels (int):
             The number of channels in the input images (defaults to 3 for RGB).
        layer_norm_eps (float):
             A small constant to prevent division by zero in layer normalization (defaults to 1e-06).
        Class Methods:
        from_dict:
             Creates a VisionConfig instance from a dictionary of parameters, filtering out any
            unrelated keys. Ensures that only attributes defined for the VisionConfig class are set.

    """

    model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    patch_size: int
    projection_dim: int
    image_size: int = 224
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, params):
        """
        Creates a new instance of the class from a dictionary of parameters.
        This class method facilitates the initialization of an instance using a dictionary where keys correspond to the names of parameters expected by the class constructor. It filters out any keys that are not recognized as valid parameters for the constructor, thus ensuring that only relevant data is used to create the new object.

        Args:
            params (dict):
                 A dictionary where keys are parameter names and values are the corresponding values to be passed to the class constructor.

        Returns:
            An instance of the cls, initialized with the parameters provided in params dict that match the class constructor signature.

        Raises:
            TypeError:
                 If any keys in the params do not correspond to the constructor's expected parameters.

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
    Checks if the input array adheres to specific shape criteria for a 4D tensor.
    This function verifies if the input array is a 4D tensor and whether its shape conforms to certain conditions. It first checks if the array has exactly four dimensions. Then, it checks if the first dimension (usually representing output channels) is the largest among them, and also if the second and third dimensions (usually representing the height and width of a kernel) are equal.

    Args:
        arr (numpy.ndarray):
             A numpy array of any shape.

    Returns:
        (bool):
             True if the array is 4D with the largest dimension first, followed by two equal dimensions, otherwise False.

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
    A class that implements a multi-head attention mechanism based on the Transformer model. It maps a query and a set of key-value pairs to an output where the query, keys, values, and output are all tensors of dimensions specified during initialization. The attention mechanism allows the model to focus on different parts of the input sequence when producing the output sequence, which is particularly useful in sequence-to-sequence tasks.

    Args:
        dims (int):
             The total number of dimensions for each attention head. Must be divisible by num_heads.
        num_heads (int):
             The number of attention heads.
        query_input_dims (Optional[int], optional):
             The number of dimensions for the query input. Defaults to None, in which case it falls back to 'dims'.
        key_input_dims (Optional[int], optional):
             The number of dimensions for the key input. Defaults to None, in which case it falls back to 'dims'.
        value_input_dims (Optional[int], optional):
             The number of dimensions for the value input. Defaults to None, in which case it falls back to 'key_input_dims'.
        value_dims (Optional[int], optional):
             The number of dimensions for the value vectors before projecting to the output. Defaults to None, in which case it falls back to 'dims'.
        value_output_dims (Optional[int], optional):
             The number of dimensions for the projected output values. Defaults to None, in which case it falls back to 'dims'.
        bias (bool, optional):
             Whether to include bias terms in the linear projection layers. Defaults to True.

    Raises:
        ValueError:
             If the 'dims' is not divisible by 'num_heads'.
            The attention mechanism computes the scaled dot-product attention over queries, keys, and values. The queries and keys are created by applying linear transformations to the input tensor 'x', and their scaled dot-product defines the attention that is used to weight the values. The output of the attention mechanism is then linearly transformed before being returned.

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
        bias: bool = True,
    ):
        """
        Initializes a multi-head attention layer with specified dimensions and number of heads.

        Args:
            dims (int):
                 The dimension of the query, key, and value vectors.
            num_heads (int):
                 The number of attention heads.
            query_input_dims (Optional[int], optional):
                 The dimension of the query input. Defaults to 'dims' if not provided.
            key_input_dims (Optional[int], optional):
                 The dimension of the key input. Defaults to 'dims' if not provided.
            value_input_dims (Optional[int], optional):
                 The dimension of the value input. Defaults to 'dims' if not provided.
            value_dims (Optional[int], optional):
                 The dimension of the value vectors before projection. Defaults to 'dims' if not provided.
            value_output_dims (Optional[int], optional):
                 The dimension of the value vectors after projection. Defaults to 'dims' if not provided.
            bias (bool, optional):
                 Whether or not to include a bias term in the linear transformations. Defaults to True.

        Raises:
            ValueError:
                 If 'dims' is not divisible by 'num_heads', indicating that the dimensions cannot be evenly divided among the heads.

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

    def __call__(self, x, mask=None):
        """
        Performs a forward pass of the multi-head attention mechanism.
        This method takes an input tensor `x` and an optional mask `mask`, applies
        learned linear projections to convert `x` into query, key, and value
        representations, and then performs scaled dot-product attention.

        Args:
            x (Tensor):
                 The input tensor of shape (batch_size, sequence_length, features).
            mask (Tensor, optional):
                 An optional mask to prevent attention to certain positions.

        Returns:
            (Tensor):
                 The output tensor after applying multi-head attention, with the same
                shape as the input tensor `x`.

        Raises:
            ValueError:
                 If the input shapes are incompatible with the expected format.

        """
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

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


class FastGELUActivation(nn.Module):
    """
    A PyTorch Module implementing the FastGELU activation function.
    This class provides a fast approximation to the Gaussian Error Linear Unit (GELU) activation
    function commonly used in neural network architectures. The fast version of the GELU uses
    a tanh approximation to the original function for a balance between performance and
    accuracy. It takes an input tensor and applies the activation function element-wise.

    Attributes:
        None

    Methods:
        __call__(input:
             mx.array) -> mx.array:
            Apply the FastGELU activation function to the input array.
        The method computes the following operation:
            0.5 * input * (1.0 + tanh(sqrt(2 / pi) * (input + 0.044715 * input ** 3)))
            where the tanh function and the square root are element-wise operations on the
            input tensor, providing a non-linear transformation typically used in hidden
            layers of neural networks.

    Args:
        input (mx.array):
             The input tensor to which the activation function will
            be applied.

    Returns:
        (mx.array):
             The output tensor with the FastGELU activation applied.

    """

    def __call__(self, input: mx.array) -> mx.array:
        """
        Performs a transformation on the given input using the GELU activation function.
        This method implements the Gaussian Error Linear Unit (GELU) activation function, which is
        used as a non-linear transformation in neural networks. It takes an input tensor and returns
        the result of the GELU function applied element-wise to that tensor. The GELU function is
        derived from the Gaussian distribution and is known for its use in transformers and other
        state-of-the-art models. It is similar to the sigmoid but with a heavier tail, allowing for more
        flexible transformations of the input data.

        Args:
            input (mx.array):
                 The input tensor to which the GELu activation function will be applied.

        Returns:
            (mx.array):
                 The result of applying the GELU activation function to the `input` tensor.

        """
        return (
            0.5
            * input
            * (1.0 + mx.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * (input**3))))
        )


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) module implementing a simple fully connected neural network architecture.
    This class inherits from `nn.Module` and constitutes a simple two-layer feedforward neural network
    with an activation function inserted between the layers. It is commonly used within larger
    configurations of neural networks for vision-related tasks.

    Attributes:
        activation_fn:
             An instance of FastGELUActivation, which applies the Gaussian Error Linear
            Unit (GELU) activation function.
        fc1 (nn.Linear):
             The first linear transformation layer, which maps input features
            from the hidden size as specified in `config` to an intermediate size.
        fc2 (nn.Linear):
             The second linear transformation layer, which maps the representation
            from the intermediate size back to the hidden size as specified in `config`.

    Args:
        config (VisionConfig):
             A configuration object containing parameters like hidden_size
            and intermediate_size which are used to define the dimensions of the linear layers.

    Note:
        The `__call__` method allows instances of this class to be used as if they were functions;
        it takes an input tensor 'x', applies the first linear transformation followed by the
        activation function, and then applies the second linear transformation, returning the output.

    Methods:
        __call__(x:
             mx.array) -> mx.array: Applies the MLP to an input tensor 'x' and returns the output.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of a neural network module.
        This initialization method sets up the neural network with a FastGELU activation function and two fully
        connected (fc) linear layers. The sizes of the layers are determined by configuration parameters provided in the `config` argument.

        Args:
            config (VisionConfig):
                 An instance of VisionConfig which provides configurations like hidden_size and
                intermediate_size that define the architecture of the neural network.

        Raises:
            TypeError:
                 If the config provided is not an instance of VisionConfig.

        """
        super().__init__()
        self.activation_fn = FastGELUActivation()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Performs a forward pass through the model.

        Args:
            x (mx.array):
                 The input array to the network.

        Returns:
            (mx.array):
                 The output of the network after passing through two fully connected layers
                and an activation function.

        Raises:
            None

        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    A layer that represents a single block in an encoder architecture, typically used in transformer-based neural networks.
    This class is responsible for processing inputs through a self-attention mechanism followed by layer normalization and a multilayer perceptron (MLP). It applies the described operations sequentially on the input data and uses residual connections after each main block (self-attention and MLP) to facilitate the training of deep networks.

    Attributes:
        embed_dim (int):
             The size of the hidden layer embeddings.
        self_attn (Attention):
             The self-attention module that performs attention operations.
        layer_norm1 (nn.LayerNorm):
             The first layer normalization module applied before the self-attention block.
        mlp (MLP):
             The multilayer perceptron that processes the output of the self-attention block.
        layer_norm2 (nn.LayerNorm):
             The second layer normalization module applied before the MLP block.

    Methods:
        __call__(x:
             mx.array, mask: Optional[mx.array] = None) -> mx.array:
            Processes the input tensor `x` using self-attention and MLP blocks with respective layer normalization and residual connections. An optional mask can be provided to exclude certain positions from attention calculations.

    Args:
        x (mx.array):
             The input tensor to the encoder layer.
        mask (Optional[mx.array]):
             An optional mask tensor for the self-attention operation.

    Returns:
        (mx.array):
             The output tensor after processing by the encoder layer.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the Vision Transformer Block.
        This method sets up the components of the Vision Transformer block, which
        include the attention mechanism, multilayer perceptron (MLP), and layer
        normalization steps. The parameters for the attention and MLP are determined by
        the provided VisionConfig object.

        Args:
            config (VisionConfig):
                 A configuration object containing parameters for
                constructing the various submodules of the Vision Transformer block,
                such as hidden_size for the dimensions of embeddings, the number of
                attention heads, layer normalization epsilon values, and sizes for
                intermediate layers in the MLP.

        Raises:
            ValueError:
                 An error occurs if the input feature dimensions in config
                are not divisible by the number of attention heads.

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
        Performs a forward pass of a transformer block on an input sequence.

        Args:
            x (mx.array):
                 The input tensor representing a sequence of tokens.
                This tensor typically has a shape (batch_size, sequence_length, embedding_dim).
            mask (Optional[mx.array]):
                 The optional mask tensor for attention. The mask specifies which
                positions are valid for attention and which are to be masked out.
                Typically, this tensor has a shape (batch_size, sequence_length).
                Defaults to None, which means no masking is applied.

        Returns:
            (mx.array):
                 The output tensor after applying self-attention and a feedforward neural
                network (MLP) to the input sequence. The output tensor has the same size
                as the input tensor 'x'.

        Raises:
            This function does not explicitly raise any exceptions. However, exceptions can occur
            internally depending on the implementation details of 'self_attn', 'layer_norm1',
            'layer_norm2', and 'mlp' if input shapes or types are incompatible with expected shapes
            or types by these components of the transformer block.


        """
        r = self.self_attn(self.layer_norm1(x), mask)
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r


class Encoder(nn.Module):
    """
    A module representing an Encoder component used in neural networks, specifically designed for vision-related tasks.
    The Encoder is responsible for transforming input data through a series of EncoderLayers, with the ability to optionally
    return all intermediate hidden states.

    Attributes:
        layers (list of EncoderLayer):
             A list containing EncoderLayer instances, each representing a layer of the encoder.

    Args:
        config (VisionConfig):
             A configuration object containing parameters for the encoder and its layers.

    Methods:
        __call__(self, x, output_hidden_states, mask):
            Processes the input data through the encoder layers and returns the final encoder states.

    Args:
        x (mx.array):
             The input data array to be passed through the encoder.
        output_hidden_states (Optional[bool]):
             If set to True, the encoder will return all intermediate hidden states.
            Defaults to None, in which case only the final hidden state is returned.
        mask (Optional[mx.array]):
             An optional mask array to be applied to the input data.
            This can be used for attention mechanisms or to mask out certain parts of the input.
            Defaults to None.

    Returns:
        (mx.array):
             If output_hidden_states is False or None, returns the last encoder state.
        tuple (mx.array, tuple of mx.arrays):
             If output_hidden_state is True,
            returns a tuple with the last encoder state and a tuple of all encoder states.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the Encoder with a specified configuration.

        Args:
            config (VisionConfig):
                 An object containing the configuration settings for the encoder.

        Raises:
            TypeError:
                 If config is not an instance of VisionConfig.

        """
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Performs a forward pass through the model layers with optional generation of encoder states.
        This method applies each layer of the encoder to the input sequence `x`, optionally accumulates encoder states,
        and returns tuple of the output of the first element of the final sequence `h` and the encoder states if requested.

        Args:
            x (mx.array):
                 The input data, where `x` is expected to be a multi-dimensional array of the shape suitable for
                the encoder.
            output_hidden_states (Optional[bool]):
                 A flag to control whether the encoder states are returned. If true,
                all the hidden states are returned, otherwise, none are returned.
            mask (Optional[mx.array]):
                 An optional mask array to be applied to the input data. This is used to mask
                out certain positions from attention calculations.

        Returns:
            (Tuple[mx.array, Optional[List[mx.array]]]):
                 A tuple containing the transformed first element of the final
                sequence as the first item, and optionally, a list of encoder
                states as the second item if `output_hidden_states` is true. If
                `output_hidden_states` is false, the second item of the tuple
                will be `None`.

        """
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for l in self.layers:
            x = l(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

            h = x[0]

        return (h, encoder_states)


class VisionEmbeddings(nn.Module):
    """
    Class representing the embedding layer used in a vision model.
    This class handles the creation of embeddings from input images. It splits images into patches and
    then embeds them using a convolution operation to create patch embeddings. Additionally, it applies
    positional embeddings to the patch embeddings. The class works with images that have been preprocessed
    to have a specific number of channels, defined by the given configuration.

    Attributes:
        config (VisionConfig):
             Configuration object containing the parameters for the vision model, including
            the number of channels in the image, hidden size of the embeddings, image size, and patch size.
        embed_dim (int):
             The size of the hidden representations (embeddings).
        image-size (int):
             The height and width of the images that the model will accept.
        patch_size (int):
             The height and width of the patches the images will be split into.
        patch_embedding (nn.Conv2d):
             Convolutional layer that creates embeddings for image patches.
        num_patches (int):
             Total number of patches that an image will be split into, based on the image and
            patch sizes.
        num_positions (int):
             Total number of position indices for positional embeddings, equal to num_patches.
        position_embedding (nn.Embedding):
             Embedding layer that assigns a unique embedding to each patch position.

    Methods:
        __call__(x:
             mx.array) -> mx.array:
            Method for forwarding an input image through the patch and position embedding layers to obtain
            the final embeddings that will be provided to the model.

    Args:
        x (mx.array):
             Tensor containing a batch of images to be embedded.

    Returns:
        (mx.array):
             A tensor containing the embedded representations of the images.


    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the Vision Transformer model with the given configuration.

        Args:
            config (VisionConfig):
                 The VisionConfig object containing model configuration parameters.

        Raises:
            ValueError:
                 If the image size is not divisible by the patch size, which is necessary to compute the number of patches consistently.

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
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Generates the embeddings for a given tensor input.
        This method takes an input tensor 'x' and first obtains patch embeddings
        from it using a 'patch_embedding' layer. Afterwards, it flattens the
        patch embeddings. Positional embeddings are then added to these
        patch embeddings based on positional IDs calculated from the number
        of positions 'num_positions'. The method returns the final embeddings
        which is a combination of patch and positional embeddings.

        Args:
            x (mx.array):
                 An input tensor that represents the raw image patches.

        Returns:
            (mx.array):
                 The final embeddings tensor after adding positional embeddings
                to patch embeddings.

        Raises:
            TypeError:
                 If the input 'x' is not of the expected type mx.array.

        """
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings
        embeddings += self.position_embedding(position_ids)
        return embeddings


class SigLipVisionModel(nn.Module):
    """
    A model class that serves as a vision model which encapsulates a series of operations typically used in a vision-based machine learning architecture. The model follows a structure where input data is first passed through an embeddings layer, then through an encoder, and finally normalized by a post-layer normalization layer. This class is built upon PyTorch's nn.Module, making it compatible with PyTorch's ecosystem of components and methodologies.

    Attributes:
        embeddings (VisionEmbeddings):
             An instance of VisionEmbeddings used to convert input data to a suitable format for the encoder.
        encoder (Encoder):
             An instance of Encoder used to encode the input embeddings into a higher-level representation.
        post_layernorm (nn.LayerNorm):
             A layer normalization that is applied to the output of the encoder.

    Args:
        config (VisionConfig):
             A configuration object containing parameters for initializing the components of the model.

    Methods:
        __call__(self, x:
             mx.array, output_hidden_states: Optional[bool]=None) -> Tuple[mx.array]:
            Forward pass of the SigLipVisionIt takes an input array and an optional argument to output hidden states. This method applies the embeddings, encoder, and post-layer normalization sequentially to the input and returns a tuple containing the pooled output, the embeddings, and the last hidden states of the encoder.

    Args:
        x (mx.array):
             The input array of visual data.
        output_hidden_states (Optional[bool]):
             A flag to determine whether or not to return the hidden states. Defaults to None.

    Returns:
        (Tuple[mx.array]):
             A tuple containing the pooled encoder output, the original embeddings, and the encoder's last hidden state (if output_hidden_states is True).

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the model with specified configuration.
        This constructor sets up the model with the necessary components based on the provided configuration.
        It initializes the embedding layer, the encoder stack, and applies layer normalization after the encoder.

        Args:
            config (VisionConfig):
                 The configuration instance containing model hyperparameters and settings.

        Attributes:
            embeddings (VisionEmbeddings):
                 The embedding layer that processes the input.
            encoder (Encoder):
                 The encoder stack consisting of multiple transformer layers.
            post_layernorm (nn.LayerNorm):
                 Normalizes the output of the encoder.


        """
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        """
        Calls the model on input data x and returns the output tensor along with auxiliary information.
        The input `x` is first passed through an embedding layer, and then the encoded output
        is generated from the encoder. Post-layernorm is applied to the first output from
        the encoder, which is considered as the pooled output of the model. Optionally,
        the encoder can be configured to return hidden states.

        Args:
            x (mx.array):
                 The input data to the model.
            output_hidden_states (Optional[bool]):
                 A flag to determine whether hidden states
                should be returned by the encoder. Defaults to None and is optional.

        Returns:
            (Tuple[mx.array]):
                 A tuple containing pooler_output, x, and encoder hidden states if
                output_hidden_states was True.

        Raises:
            This function does not raise any specific exceptions but expects the inputs to be
            correctly formatted and the required libraries to be imported previously.

        """
        x = self.embeddings(x)

        encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )

        pooler_output = self.post_layernorm(encoder_outputs[0])

        return pooler_output, x, encoder_outputs[-1]


class VisionModel(nn.Module):
    """
    A PyTorch module for a vision model with configuration checks and sanitization.

    Attributes:
        model_type (str):
             The type of model, which should be 'siglip_vision_model'.
        vision_model (SigLipVisionModel):
             An instance of the SigLipVisionModel.

    Methods:
        __init__(self, config:
             VisionConfig): Initializes the VisionModel with the given configuration.

    Args:
        config (VisionConfig):
             The configuration settings for the model.

    Raises:
        ValueError:
             If the 'model_type' in the config is not supported.
        __call__(self, x:
             mx.array, output_hidden_states: Optional[bool]=None) -> mx.array:

    Args:
        x (mx.array):
             The input data to the vision model.
        output_hidden_states (Optional[bool]):
             Flag to determine whether to output hidden states.

    Returns:
        (mx.array):
             The output from the vision model.
        sanitize(self, weights):
            Processes the weights to ensure they are in the correct format.

    Args:
        weights (dict):
             A dictionary containing the weights of the model.

    Returns:
        (dict):
             A dictionary with the sanitized weights.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the model with the given configuration.

        Args:
            config (VisionConfig):
                 Configuration instance that contains model-specific parameters.

        Raises:
            ValueError:
                 If the 'model_type' attribute of the config is not 'siglip_vision_model',
                indicating an unsupported model type.

        Attributes:
            model_type (str):
                 A string indicating the type of the model.
            vision_model (SigLipVisionDBModel):
                 An instance of SigLipVisionModel.

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
        Performs a forward pass of the vision model.
        This method wraps around a pre-defined vision model to process the input data
        and optionally return the hidden states.

        Args:
            x (mx.array):
                 The input array to the vision model.
            output_hidden_states (Optional[bool], None):
                 A flag to determine whether the hidden states should be
                returned or not. Defaults to None, in which case the default behavior of the vision model's forward pass
                is used.

        Returns:
            (mx.array):
                 The output of the vision model, which could be the final predicted values or
                a tuple including the hidden states depending on the `output_hidden_states` flag.


        """
        return self.vision_model(x, output_hidden_states)

    def sanitize(self, weights):
        """
        Definitely sanitizes a dictionary of weights by removing specific keys and adjusting the
        shape of tensor weights for patch embeddings.
        This method iterates through the dictionary of weight tensors and performs the
        following actions:
        - Excludes any weights associated with the key that contains 'position_ids',
        as these are not to be sanitized.
        - Adjusts tensor shapes for keys that include 'patch_embedding.weight'. It ensures
        that the weight tensor is in the desired shape by checking its dimensions with
        the 'check_array_shape' function. If the shape is incorrect, it transposes the
        tensor to have the correct orientation (typically arranging the out_channels to
        be the first dimension).
        - Retains all other weights without modification.

        Args:
            weights (dict):
                 A dictionary where keys are strings representing the weight
                names and values are the weight tensors needing
                sanitation.

        Returns:
            (dict):
                 A new dictionary with the sanitized weights.

        Raises:
            ValueError:
                 If the 'check_array_shape' indicates that the shape of
                'patch_embedding.weight' does not meet the required criteria and
                cannot be corrected by a simple transpose operation. This exception
                raise is implicit, as it is raised within the 'check_array_shape'
                helper function which is called within this method.

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
