"""

The `vision` module encapsulates a complete framework for building vision models, particularly tailored for tasks involving image processing and analysis. The module includes a hierarchy of classes and methods designed to represent components of neural networks and their related configurations. Below is a systematic description of each class and function contained within the module, with their purpose, structure, and interactions.

### VisionConfig
- **Purpose**: Represents the configuration options required to define an architecture for a vision model, including attributes such as model type, layer sizes, and dimensions.
- **Structure**: This class is a simple dataclass that holds configuration attributes as fields. It supports instantiation from a dictionary of parameters through the `from_dict` class method.

### check_array_shape
- **Purpose**: Validat...
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
    A simple data class that encapsulates configuration parameters for a vision model.

    Attributes:
        model_type (str):
             The type of vision model (e.g., 'CNN', 'ResNet', 'Transformer').
        hidden_size (int):
             The size of the hidden layers within the model.
        intermediate_size (int):
             The size of the intermediary layer if applicable.
        num_hidden_layers (int):
             The number of hidden layers present in the model.
        num_attention_heads (int):
             The number of attention heads in the model (relevant for models like Vision Transformer).
        image_size (int):
             The size of the image input that the model expects in terms of pixel width and height (assumes square images).
        patch_size (int):
             The size of the patches the image is divided into (relevant for patch-based approaches like Vision Transformer).
        layer_norm_eps (float):
             A small value added to the denominator in the layer normalization equation to improve numerical stability.
        num_channels (int, optional):
             The number of channels in the image inputs (default is 3 for RGB images).
        Class Methods:
        from_dict(cls, params):
             A class method that creates an instance of VisionConfig from a dictionary of parameters.
            Ensures only parameters that are valid for VisionConfig instantiation are used.

    Args:
        params (dict):
             A dictionary where keys represent parameter names and values are the corresponding parameter values.

    Returns:
        An instance of VisionConfig constructed with the named parameters from the provided dictionary.


    """

    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float
    num_channels: int = 3

    @classmethod
    def from_dict(cls, params):
        """
        Constructs an instance of the class from a dictionary of parameters.
        This method initializes a new instance of the class using only the keys that are valid
        parameters for the class constructor. Invalid keys will be ignored.

        Args:
            params (Dict[str, Any]):
                 A dictionary where keys are the names of the parameters
                expected by the class constructor and values are the values to be set for those
                parameters.

        Returns:
            An instance of the class populated with values from the `params` dictionary, provided
            the keys in `params` match the constructor's expected parameters.

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
    Checks if the provided array has a valid shape for convolutional kernel filters.
    This function verifies that the input array adheres to certain constraints that are typical for convolutional
    kernel filters in neural network architectures. Specifically, the array must have four dimensions, where the
    number of output channels is the largest dimension, and the height and width of the kernel (filter size)
    must be equal.

    Args:
        arr (ndarray):
             The array to be checked for proper shape.

    Returns:
        (bool):
             True if the array shape is valid according to the specified rules, False otherwise.

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
    A customizable multi-headed attention module for neural networks.
    This module implements a multi-headed attention mechanism as a subclass of `nn.Module`. It allows for
    different input dimensions for queries, keys, and values, and provides the option to specify the number of heads
    for the attention mechanism. The feature dimensions must be divisible by the number of heads.

    Attributes:
        num_heads (int):
             The number of attention heads.
        q_proj (nn.Linear):
             Linear projection layer for queries.
        k_proj (nn.Linear):
             Linear projection layer for keys.
        v_proj (nn.Linear):
             Linear projection layer for values.
        out_proj (nn.Linear):
             Linear projection layer for the output values.
        scale (float):
             Scaling factor for the attention scores (derived from head dimensions).

    Args:
        dims (int):
             The dimensionality of the input features.
        num_heads (int):
             The number of attention heads to use.
        query_input_dims (Optional[int]):
             The dimensionality of the query input. Defaults to `dims` if not given.
        key_input_dims (Optional[int]):
             The dimensionality of the key input. Also determines `value_input_dims` if it is not given. Defaults to `dims` if not given.
        value_input_dims (Optional[int]):
             The dimensionality of the value input. Defaults to `key_input_dims` if not given.
        value_dims (Optional[int]):
             The dimensions of values before the output projection. Defaults to `dims` if not given.
        value_output_dims (Optional[int]):
             The dimensions of values after the output projection. Defaults to `dims` if not given.

    Raises:
        ValueError:
             If `dims` is not divisible by `num_heads`.

    Note:
        The `__call__` method is used to forward an input tensor through the attention mechanism and it
        should not be called directly; instead, the instance of this class should be called as if it were a function.

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
    ):
        """
        Initializes a self-attention module with customizable input and output dimensions for query, key, value projections, and the number of attention heads.

        Parameters:
            dims (int):
                 The dimensionality of the feature space.
            num_heads (int):
                 The number of attention heads to use.
            query_input_dims (Optional[int]):
                 The dimensionality of the query input feature space. Defaults to `dims`.
            key_input_dims (Optional[int]):
                 The dimensionality of the key input feature space. Defaults to `dims`.
            value_input_dims (Optional[int]):
                 The dimensionality of the value input feature space. Defaults to the value of `key_input_dims`.
            value_dims (Optional[int]):
                 The dimensionality of the value feature space. Defaults to `dims`.
            value_output_dims (Optional[int]):
                 The projection dimensionality of the output from the attention operation. Defaults to `dims`.

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

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=True)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=True)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=True)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=True)

    def __call__(self, x: mx.array, mask=None):
        """
        Performs a forward pass of the multi-head attention mechanism on input tensors.
        This method applies a multi-head attention mechanism to project the input `x`
        into query, key, and value tensors, and then computes the scaled dot-product
        attention. The attention output is then projected back to the original dimension
        using an output projection layer. This method supports an optional mask.

        Args:
            x (mx.array):
                 A 3D tensor with shape (batch_size, sequence_length, feature_size)
                representing the input features to the attention layer.
            mask (optional):
                 An optional mask tensor to apply to attention scores before
                softmax normalization. The mask can be used to ignore certain positions
                during attention. Default is None.

        Returns:
            A 3D tensor with shape (batch_size, sequence_length, feature_size) after
            applying multi-head attention with the provided mask (if any).

        Raises:
            MXNetError:
                 If an MXNet-specific error occurs during the computation of
                the scaled dot product attention or any other operation within the method.

        """
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads

        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) class that extends `nn.Module`.
    This class implements a basic two-layer neural network using a specified activation function
    between the layers. It can be configured with different sized layers matching the provided
    configuration.

    Attributes:
        activation_fn:
             An instance of `nn.GELU` defining the activation function with 'fast' approximation.
        fc1:
             The first `nn.Linear` layer that transforms the input dimensions to an intermediate size.
        fc2:
             The second `nn.Linear` layer that maps the intermediate representation back to the hidden size.

    Args:
        config (VisionConfig):
             A configuration object containing model hyperparameters such as hidden size
            and intermediate size for the linear layers.

    Methods:
        __call__(x:
             mx.array) -> mx.array:
            A method that allows the object to be called as a function, applying the operations of the MLP
            network to the input tensor `x` and returning the output tensor.

    Args:
        x (mx.array):
             A tensor containing the input data to the MLP.

    Returns:
        (mx.array):
             The result of the MLP forward pass, which is the transformed input tensor after applying
            both linear layers and the activation function.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the network with the provided configuration.

        Args:
            config (VisionConfig):
                 A configuration object containing the parameters for network initialization, such as hidden_size and intermediate_size.

        Raises:
            TypeError:
                 If the passed config is not an instance of VisionConfig.

        """
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Performs a forward pass on the input data using the model's architecture.

        Args:
            x (mx.array):
                 The input data to the model as a MXNet array.

        Returns:
            (mx.array):
                 The output of the model after passing through two fully connected layers and an activation function.


        """
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    A single layer of an encoder architecture that applies self-attention and feed-forward neural network operations.
    This class represents a single layer in an encoder stack of a transformer-based
    model architecture for vision tasks. It combines self-attention, layer normalization,
    and a multi-layer perceptron (MLP) to process input features.

    Attributes:
        embed_dim (int):
             The dimensionality of the input embeddings.
        self_attn (Attention):
             The self-attention mechanism component.
        layer_norm1 (nn.LayerNorm):
             The first layer normalization component applied before self-attention.
        mlp (MLP):
             The feed-forward neural network component.
        layer_norm2 (nn.LayerNorm):
             The second layer normalization component applied after the MLP.

    Args:
        config (VisionConfig):
             Configuration object containing parameters for
            constructing the layer components, such as hidden size, number of attention
            heads, and layer normalization epsilon value.

    Methods:
        __call__(self, x:
             mx.array, mask: Optional[mx.array]=None) -> mx.array:
            Applies the layer operations sequentially on the input `x`. First, a layer
            normalization followed by self-attention, a residual connection, another
            layer normalization, and finally the MLP operation. Optionally applies a
            mask during the self-attention step.

    Args:
        x (mx.array):
             The input features to the encoder layer.
        mask (Optional[mx.array], optional):
             An optional mask to exclude certain
            positions from attending to others. Defaults to None.

    Returns:
        (mx.array):
             The output features from the encoder layer after processing.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of the transformer block as defined in a vision-based model architecture.

        Args:
            config (VisionConfig):
                 An instance of VisionConfig class containing various configuration parameters.

        Attributes:
            embed_dim (int):
                 The size of the embedding dimension.
            self_attn (Attention):
                 An Attention module that calculates the self-attention for the input sequences.
            layer_norm1 (nn.LayerNorm):
                 The first layer normalization with the embedding dimensions and a small epsilon for numerical stability.
            mlp (MLP):
                 A multilayer perceptron module as defined by the vision model architecture.
            layer_norm2 (nn.LayerNorm):
                 The second layer normalization, applied after the MLP module.

        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Performs a forward pass on given input tensor using an architecture similar to the Transformer block.
        This method sequentially applies layer normalization, self-attention, and a multilayer perceptron (MLP) to the input tensor 'x'. The output of each module is added to the input of that module (residual connection) before further processing. Optionally, a mask can be applied to the self-attention mechanism.

        Args:
            x (mx.array):
                 The input tensor to the Transformer block.
            mask (Optional[mx.array]):
                 The mask tensor to be applied during self-attention computation. Defaults to None.

        Returns:
            (mx.array):
                 The resultant tensor after applying the Transformer block operations to the input tensor.

        Raises:
            No explicit exceptions are raised but underlying layers (e.g., layer_norm1, self_attn, mlp) might raise exceptions if the input arguments do not match their expected formats or if there is an error in computation.

        """
        y = self.layer_norm1(x)
        y = self.self_attn(y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """
    Encoder class for Vision models.
    The Encoder class is derived from nn.Module and is responsible for encapsulating
    multiple `EncoderLayer` instances. It processes input data through these layers and
    can provide access to intermediate hidden states if required.

    Attributes:
        layers (List[EncoderLayer]):
             A list of `EncoderLayer` instances which constitute
            the encoder part of the network.

    Args:
        config (VisionConfig):
             An instance of VisionConfig containing configuration
            parameters used to set up the Encoder.

    Methods:
        __call__(self, x, output_hidden_states, mask) -> Tuple[mx.array, Optional[Tuple[mx.array]]]:
            Processes the input through the encoder layers and returns the output along
            with the hidden states if `output_hidden_states` is True.

    Args:
        x (mx.array):
             The input data to be processed by the encoder.
        output_hidden_states (Optional[bool]):
             Whether to return the hidden states
            of all encoder layers. Defaults to None, if not provided, it will be
            interpreted as `False` and hidden states will not be returned.
        mask (Optional[mx.array]):
             An optional mask array to be applied to the input
            data. It's used to skip certain tokens from processing and attention
            mechanisms. Defaults to None.

    Returns:
        (mx.array):
             The output of the last encoder layer.
        (Optional[Tuple[mx.array]]):
             A tuple containing the hidden states of all
            encoder layers if `output_hidden_matched` is True, otherwise `None`.


    """

    def __init__(self, config: VisionConfig):
        """
        Initializes the encoder with the specified configuration.

        Args:
            config (VisionConfig):
                 An instance of VisionConfig that contains the configuration parameters for the
                encoder layers.
                The initialization process includes creating a series of encoder layers based on the number of hidden
                layers defined in the configuration. Each encoder layer is instantiated with the same configuration.

        Raises:
            TypeError:
                 If `config` is not an instance of VisionConfig.

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
        Process the input through the SigLipVisionModel and produce the output.
        The method takes an input tensor `x`, computes the vision embeddings, processes it through all the layers of the Encoder,
        and finally applies a layernorm and a multi-head attention pooling head to produce the pooler output. If requested,
        the hidden states of all layers can also be returned.

        Args:
            x (mx.array):
                 The input tensor containing image or other vision data to process.
            output_hidden_states (Optional[bool], optional):
                 Flag indicating whether to return the hidden states of all layers.
                If True, the hidden states are returned along with the pooler output and the last layer output. If not provided or False, it is ignored. Defaults to None.

        Returns:
            (mx.array):
                 A tuple containing the following elements:
            (- Pooler output):
                 The output of the pooling head, representing the attended combination of the sequence elements.
            (- Last layer output):
                 The output of the last layer of the encoder.
            (- Encoder_states):
                 The tuple of hidden states from each layer of the encoder. This is only returned if `output_hidden_states` is set to True.


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
    A PyTorch module for creating embeddings from image inputs. It takes a configuration object specifying the details of the model architecture. This module generates patch embeddings by partitioning the images into fixed-size patches and applies a convolution operation to create embeddings for each patch. It also handles positional embeddings for these patches to retain spatial information.

    Attributes:
        config (VisionConfig):
             The configuration object containing model specifications.
        embed_dim (int):
             The dimension of the embeddings.
        image_size (int):
             The size (height and width) of the input images.
        patch_size (int):
             The size of each image patch.
        patch_embedding (nn.Conv2d):
             A convolutional layer that creates embeddings for image patches.
        num_patches (int):
             The number of patches along one dimension of the image.
        num_positions (int):
             The total number of positions for positional embeddings, which is the square of `num_patches`.
        position_embedding (nn.Embedding):
             An embedding layer to encode positional information for each patch.

    Methods:
        __call__(self, x:
             mx.array, mask: Optional[mx.array]) -> mx.array:
            Generates embeddings for input images and applies positional embeddings.

    Args:
        x (mx.array):
             A batch of images with shape (B, H, W, C), where B is the batch size, H is the height, W is the width, and C is the number of channels.
        mask (Optional[mx.array]):
             An optional attention mask to specify which patches are valid (have a value of 1).

    Returns:
        (mx.array):
             The resulting embeddings with added positional encodings.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of the Vision model with the given configuration.
        This constructor initializes the Vision model with the specified configuration by setting up the necessary parameters and layers for image processing. The initialization includes configuring the embedding dimension, image size, and patch size according to the given configuration. Additionally, it sets up a 2D convolutional layer (patch_embedding) for embedding the image patches and a position embedding to capture the positional information of image patches in the input sequence.

        Args:
            config (VisionConfig):
                 An instance of VisionConfig which provides the hyperparameters for the model.

        Raises:
            TypeError:
                 If the provided config is not an instance of VisionConfig.

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

        self.num_patches = self.image_size // self.patch_size
        self.num_positions = self.num_patches**2
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Performs a forward pass of the model by mapping input patches to embeddings and adjusting them with positional encodings.

        Args:
            x (mx.array):
                 The input tensor representing images with shape (B, H, W, C), where B is batch size, H is height,
                W is width, and C is the number of channels.
            mask (Optional[mx.array]):
                 An optional mask tensor, with bool or binary values, indicating the valid patches for
                the positional encodings. Shape is (B, nb_patches_h, nb_patches_w), where 'nb_patches_h' and
                'nb_patches_w' are the numbers of patch partitions along the height and width axis.
                Default is None, which means no mask is applied.

        Returns:
            (mx.array):
                 The tensor with patch embeddings augmented with positional information. Shape of the returned tensor
                corresponds to (B, D), where D is the dimension of the embeddings.

        Raises:
            ValueError:
                 If `mask` is given but does not match the expected shape based on the input tensor `x` and model
                configurations like number of patches.

        """
        B, H, W, C = x.shape
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        max_nb_patches_h, max_nb_patches_w = (
            H // self.patch_size,
            W // self.patch_size,
        )
        boundaries = np.linspace(
            1 / self.num_patches, 1.0, self.num_patches, endpoint=False
        )
        position_ids = np.zeros((B, max_nb_patches_h * max_nb_patches_w), dtype=int)

        for batch_idx, p_attn_mask in enumerate(mask):
            p_attn_mask = np.array(p_attn_mask)
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0, :].sum()

            fractional_coords_h = np.linspace(0, 1, nb_patches_h, endpoint=False)
            fractional_coords_w = np.linspace(0, 1, nb_patches_w, endpoint=False)

            bucket_coords_h = (
                np.digitize(fractional_coords_h, boundaries, right=True) - 1
            )
            bucket_coords_w = (
                np.digitize(fractional_coords_w, boundaries, right=True) - 1
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.reshape(-1)] = pos_ids

        embeddings = patch_embeddings
        embeddings += self.position_embedding(mx.array(position_ids))
        return embeddings


class VisionModel(nn.Module):
    """
    A PyTorch module representing a vision model for image classification tasks.
    This class is a PyTorch Module subclass implementing a VisionModel. It has a specific architecture defined
    by the provided configuration object of type VisionConfig. The VisionModel only supports the 'idefics2'
    model type and will raise an exception for any other type. It is composed of embeddings for processing
    input image patches, an encoder, and a post-layer normalization layer.

    Attributes:
        config (VisionConfig):
             A configuration object containing model parameters such as model type, hidden size,
            patch size, and other relevant hyperparameters.
        model_type (str):
             Type of model, currently restricted to 'idefics2'.
        embeddings (VisionEmbeddings):
             Embeddings layer for processing the input.
        encoder (Encoder):
             Encoder layer for transforming embeddings.
        post_layernorm (nn.LayerNorm):
             Post-encoder layer normalization.

    Methods:
        __call__(self, x, patch_attention_mask=None, output_hidden_states=None):
            Performs a forward pass on the input tensor 'x'. Optionally accepts atensor for patch attention mask
            and a boolean to determine whether to output hidden states.
        sanitize(self, weights):
            Sanitizes the input weights to ensure compatibility with the model’s patch embedding weights.

    Raises:
        ValueError:
             If the given model type in 'config' is not 'idefics2'.

    """

    def __init__(self, config: VisionConfig):
        """
        Initializes a new instance of a deep learning model with a specific configuration.

        Args:
            config (VisionConfig):
                 An instance of the VisionConfig class that contains configuration parameters for the model.

        Raises:
            ValueError:
                 If the model_type attribute within config is not 'idefics2', indicating an unsupported model type.

        """
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type != "idefics2":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        """
        Performs a forward pass on the input array with the model's defined architecture.
        This method applies embedding transformations on the input, processes it through the encoder, and then normalizes
        the encoder's output. It is designed as a callable for a class that handles transformer-based architectures, which
        could be useful within models that process images or patches of images through attention mechanisms.

        Args:
            x (mx.array):
                 The input feature array of shape (Batch size, Sequence length, Dimension, Channels).
            patch_attention_mask (Optional[mx.array]):
                 A boolean array that defines the attention mask for patches.
                If not provided, it is assumed that all patches are to be attended to equally. The default mask is created based
                on the patch size defined in the class configuration. The shape should be (Batch size, Sequence length // patch size,
                Dimension // patch size). Defaults to None.
            output_hidden_states (Optional[bool]):
                 A flag that indicates whether the hidden states from all layers should be
                returned. Defaults to None, which means only the final layer's hidden states are returned.

        Returns:
            (tuple):
                 A tuple containing:
            - pooler_output (mx.array):
                 The normalized output from the encoder's first token.
            - x (mx.array):
                 The embedded input feature array.
            - hidden_states (mx.array or None):
                 The hidden states from all encoder layers if output_hidden_states is True;
                otherwise, None.

        """
        B, L, D, C = x.shape
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = mx.ones(
                (
                    B,
                    L // patch_size,
                    D // patch_size,
                ),
                dtype=mx.bool_,
            )

        x = self.embeddings(x, mask=patch_attention_mask)

        encoder_outputs = self.encoder(x=x, output_hidden_states=output_hidden_states)

        pooler_output = self.post_layernorm(encoder_outputs[0])

        return pooler_output, x, encoder_outputs[-1]

    def sanitize(self, weights):
        """
        Sanitizes the given weights dictionary by checking and adjusting the shape of specific weight tensors.
        The function iterates through the key-value pairs in the weights dictionary. If the key contains 'patch_embedding.weight' and the corresponding weight tensor does not have the expected shape, it is transposed. The expected shape is such that the number of output channels (first dimension) should be the largest, and the height (second dimension) and width (third dimension) should be equal. If the shape is already correct, the original tensor is kept unchanged. All other weight tensors are included in the sanitized output as they are.

        Args:
            weights (dict):
                 A dictionary containing the model weights as key-value pairs, where the keys are string identifiers for the weights, and the values are the weight tensors.

        Returns:
            (dict):
                 A dictionary containing the sanitized weights. Specifically, tensors at keys with 'patch_embedding.weight' that do not meet the shape criteria are transposed. All other weights remain unmodified.

        """
        sanitized_weights = {}
        for k, v in weights.items():
            if "patch_embedding.weight" in k:
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
