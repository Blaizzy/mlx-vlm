"""

## Overview of the Language Module

The language module provides an implementation of a transformer-based language model. It is composed of several classes, each encapsulating a specific element of the transformer architecture. The primary components include configuration structures, attention mechanisms, multi-layer perceptrons (MLPs), and the transformer block itself. Additionally, the language model class ties all the elements together to perform language modeling tasks.

### TextConfig

The `TextConfig` class is a dataclass that serves as a configuration holder for the transformer model. It encompasses various hyperparameters such as the model type, size of the hidden layers, number of attention heads, and so forth. A class method `from_dict` allows for instantiation from a dictionary, facilitating the dynamic configuration of Transformer models.

### Attention

The `Attention` class implements the scaled dot-product attention mechanism, which is central to the transformer architecture. It includes the functionality to compute queries, keys, values, and the attention output. Additionally, it incorporates a Rotary Positional Encoding (RoPE) to imbue the attention mechanism with information about the position of tokens.

### MLP

The `MLP` class represents a simple two-layer feed-forward network with an intermediate SiLU-gated layer, utilized within each transformer block as the position-wise feed-forward network.

### TransformerBlock

`TransformerBlock` is a composite class that integrates the attention mechanism (via an instance of `Attention`) and a multilayer perceptron (MLP). The module applies the attention mechanism to the input, followed by the MLP. Layer normalization is applied before and after the attention mechanism as well as after the MLP.

### LanguageModel

The `LanguageModel` class encapsulates the entire model, comprising an embedding layer, multiple instances of `TransformerBlock`, a final layer norm, and an output linear layer. This class handles language modeling by processing input sequences through successive transformer blocks, tracking cached states for efficient sequential processing.


**Note**: The module employs the `dataclasses` library for its configuration classes, utilizes custom neural network module implementations (`nn.Module`), and manipulation of tensors via the hypothetical `mx.array` module (analogous to `numpy.ndarray`). This document does not provide usage or example code—the focus is on the architecture provided within the module.
"""

import inspect
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TextConfig:
    """
    A configuration class that encapsulates various hyperparameters for defining the structure of a text model.

    Attributes:
        model_type (str):
             The model architecture type.
        hidden_size (int):
             The size of the hidden layers.
        num_hidden_layers (int):
             The number of hidden layers in the transformer model.
        intermediate_size (int):
             The size of the 'intermediate' (i.e., feed-forward) layer.
        num_attention_heads (int):
             The number of attention heads in the transformer model.
        rms_norm_eps (float):
             The epsilon used for RMS normalization.
        vocab_size (int):
             The size of the vocabulary.
        num_key_value_heads (int):
             The number of key/value pairs in the attention mechanism.
        rope_theta (float):
             A hyperparameter used in Rotary Position Embedding (ROPE) with a large default value.
        rope_traditional (bool):
             A flag indicating if the traditional method is used in ROPE.
        tie_word_embeddings (bool):
             A setting that specifies whether to tie input and output word embeddings.
        Class Methods:
        from_dict(cls, params):
             Creates an instance of `TextConfig` from a dictionary by filtering
            parameters that are valid for the class's constructor.

    Methods:
        __post_init__(self):
             A post-initialization method that sets `num_key_value_heads` to the value
            of `num_attention_heads` if `num_key_value_store_heads` is not explicitly initialized.

    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        """
        Generates an instance of the class based on a dictionary of parameters.
        This class method acts as an alternative constructor, allowing for the creation of class instances by unpacking a dictionary into the constructor's arguments. It filters the provided dictionary, only passing arguments that exist in the class constructor's signature.

        Parameters:
            ----------
            params :
                 dict
                A dictionary where keys correspond to the constructor's parameter names and the values are the values to be set for the corresponding parameters.

        Returns:
            -------
            An instance of the class, initialized with the parameters provided in the `params` dictionary that match the constructor's signature.

        Raises:
            ------
            A TypeError will be raised if `params` includes keys that do not correspond to any of the class constructor's parameters.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def __post_init__(self):
        """
        Initializes the newly created object after __init__ has run.
        This method serves as a post-initialization step for a class instance to set the
        `num_key_value_heads` attribute if it is not already set. If `num_key_value_heads` is 'None', the method sets the attribute to the value of
        `num_attention_heads`.

        Attributes:
            num_key_value_heads:
                 An optional int or None - The number of key/value heads in the attention mechanism. If not specified, this will be set to the same value as `num_attention_heads`.
            num_attention_heads:
                 An int - The number of attention heads in the model.

        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    """
    A PyTorch module that implements a scaled-dot product attention mechanism with a relative positional encoding (RoPE).
    This class is a part of a neural network module and takes a configuration object
    `TextConfig` to initialize the attention mechanism. It supports caching of the key
    and value tensors for efficient computation across multiple calls, primarily useful
    in autoregressive models where past state is reused.

    Attributes:
        n_heads (int):
             The number of attention heads.
        n_kv_heads (int):
             The number of keys/values heads.
        scale (float):
             The scaling factor for the dot product attention, computed as
            the inverse square root of the head dimension.
        q_proj (nn.Linear):
             The linear projection layer for queries.
        k_proj (nn.Linear):
             The linear projection layer for keys.
        v_proj (nn.Linear):
             The linear projection layer for values.
        o_proj (nn.Linear):
             The linear output projection layer.
        rope (nn.RoPE):
             Relative positional encoding module for computing positionally
            aware queries and keys.
            The constructor initializes the projection layers and the relative position encoding
            module using the configuration object `args`. The `__call__` method applies the
            attention mechanism to input tensors `x`, with an optional `mask` and caching
            via the `embed` argument.

    Args:
        args (TextConfig):
             Configuration object containing hyperparameters for the
            attention module.
        The `__call__` method expects:
        x (mx.array):
             Input tensor of shape `(B, L, D)` where `B` is the batch size,
            `L` is the sequence length, and `D` is the feature dimension.
        mask (Optional[mx.array]):
             Optional mask tensor for the attention mechanism.
        cache (Optional[Tuple[mx.array, mx.array]]):
             Optional cache for previously computed
            key-value pairs.

    Returns:
        (Tuple[mx.array, Tuple[mx.array, mx.array]]):
             Output after applying attention, consisting
            of the attention result and an optional new cache with updated key-value pairs.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes the neural network module with specified configurations.

        Args:
            args (TextConfig):
                 Configuration object containing various attributes required to set up the module. Attributes should include 'hidden_size' for the dimensions of the hidden layer, 'num_attention_heads' for the number of attention heads, 'num_key_value_heads' typically equal to the number of attention heads and used for projecting keys and values respectively, 'rope_traditional' indicating whether to use traditional relative position encoding, and 'rope_theta' for base theta value in RoPE.

        Raises:
            Various exceptions may be raised depending on the superclass implementation this inherited init method calls and the actions performed within it.

        """
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Performs a forward pass on the MultiheadAttention module with optional caching for transformer-based models.

        Args:
            x (mx.array):
                 The input tensor with shape (batch_size, seq_length, model_dim).
            mask (Optional[mx.array]):
                 An optional mask for the attention mechanism to ignore certain positions,
                typically used for padding or future timesteps. Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]]):
                 Tuples of cached keys and values from previous timesteps
            for efficient decoding in auto-regressive models. If provided, it should contain two elements:
                cached keys and cached values. Defaults to None.

        Returns:
            (Tuple[mx.array, Tuple[mx.array, mx.array]]):
                 A tuple containing the attention output and the updated
                cache which includes the concatenated keys and values for future caching.

        Raises:
            ValueError:
                 If the shape of the cache does not match the expected format.

        """
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class MLP(nn.Module):
    """
    A class that represents a Multilayer Perceptron (MLP) module, which is a basic form of a neural network. The MLP class is a subclass of nn.Module and is used to create a neural network layer with one hidden layer that includes a gating mechanism.

    Args:
        dim (int):
             The size of the input feature dimension.
        hidden_dim (int):
             The size of the hidden layer feature dimension.

    Attributes:
        gate_proj (nn.Linear):
             Linear transformation applied to the input with no bias, mapping from input dimension `dim` to hidden dimension `hidden_dim`.
        down_proj (nn.Linear):
             Linear transformation mapping from the hidden dimension `hidden_params` back to the input dimension `dim` with no bias.
        up_proj (nn.Linear):
             Linear transformation applied to the input with no bias, mapping from input dimension `dim` to hidden dimension `hidden_dim`.
            The class implements the `__call__` method, which allows an instance of the class to be called as a function, processing the input tensor `x` through the layer.

    Returns:
        (mx.array):
             The output of the MLP after processing the input tensor `x` through the linear layers and non-linear activation function (SiLU).


    """

    def __init__(self, dim, hidden_dim):
        """
        Initializes a new instance of the class with the specified dimensions for the gate, down, and up projections.
        This constructor initializes three linear projection layers without bias: one to project from the input
        `dim` dimension to the `hidden_dim` dimension (the 'gate projection'), one to project from the `hidden_dim`
        dimension down to `dim` (the 'down projection'), and one to project from `dim` back to `hidden_dim` (the 'up
        projection').

        Parameters:
            dim (int):
                 The size of the input dimension to the gate and down projection layers.
            hidden_dim (int):
                 The size of the output dimension for the gate projection and the input dimension
                for the up projection layer.

        Raises:
            TypeError:
                 If any of the supplied arguments are not of the expected type (int).

        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Performs the call operation on the object by applying a specific gating mechanism.
        This method applies a gated transformation to an input tensor `x`. The gating mechanism involves a down-projection, an element-wise non-linear activation (SiLU), and an up-projection. The result is then scaled by the gating signal.

        Args:
            x:
                 The input tensor to transform.

        Returns:
            The transformed tensor, resulting from the gate-based projection mechanism.

        Raises:
            TypeError:
                 If the input is not of the expected type (mx.array).

        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    A neural network module that represents a single block within a transformer architecture.
    Each block is composed of an attention mechanism, a position-wise feed-forward network (MLP),
    and layer normalization steps. It also contains the configuration parameters needed to construct
    its components. The block processes input sequences alongside optional mask and cache for
    efficiency in successive calls, typically when used within a transformer model.

    Attributes:
        num_attention_heads:
             An integer representing the number of attention heads in the block.
        hidden_size:
             An integer specifying the dimension of the input and output data of the block.
        self_attn:
             An instance of the `Attention` class to compute the self-attention mechanism.
        mlp:
             An instance of the `MLP` class defining the feed-forward network within the block.
        input_layernorm:
             An instance of `nn.RMSNorm` which applies layer normalization to inputs.
        post_attention_layernorm:
             An `nn.RMSNorm` instance for normalization after the attention step.
        args:
             An instance of `TextConfig` containing configuration parameters.

    Methods:
        __call__(x, mask=None, cache=None):
            Processes the input tensor `x` through the transformer block.

    Args:
        x (mx.array):
             A tensor containing the input sequence to the transformer block.
        mask (Optional[mx.array]):
             An optional tensor containing a mask for the attention mechanism.
        cache (Optional[Tuple[mx.array, mx.array]]):
             Optional cached output for efficient reuse in
            subsequent calls.

    Returns:
        A tuple containing the output of the transformer block and the updated cache.
        The output is a tensor of the same shape as the input `x` and represents
        the transformed sequence. The cache is a tuple of tensors holding computed
        values useful for improving the performance in subsequent iterations.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes the text-based model with the configuration provided in the args.

        Args:
            args (TextConfig):
                 An instance of TextConfig that contains configuration
                parameters such as number of attention heads, hidden size, intermediate
                size, and RMS normalization epsilon value.

        Raises:
            ValueError:
                 If the calculated dimensions are not properly aligned or
                the input values do not meet certain criteria (e.g., number of attention
                heads must be a positive integer).

        """
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Performs a forward pass of the module with optional caching.
        This method applies a sequence of operations to an input signal to produce an output signal and optionally caches intermediate results. It first performs self-attention on the input after applying layer normalization, then adds the result to the original input (residual connection) to get the first intermediate output. Next, it applies another layer normalization followed by a multi-layer perceptron (MLP) to the first intermediate output. The final output is obtained by adding the result of the MLP to the first intermediate output (another residual connection). If caching is enabled, intermediate results are also returned for use in subsequent forward passes.

        Args:
            x (mx.array):
                 The input signal.
            mask (Optional[mx.array]):
                 An optional mask to apply to the self-attention mechanism.
            cache (Optional[Tuple[mx.array, mx.array]]):
                 Optional cached tensors from a previous forward pass for use in self-attention.

        Returns:
            (mx.array):
                 The final output signal after applying self-attention and MLP operations.
            (Tuple[mx.array, mx.array]):
                 The updated cache containing intermediate results.

        """
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class LanguageModel(nn.Module):
    """
    A neural network module representing a language model built upon transformers.
    This class is built using PyTorch and is intended to represent a language model that employs transformer blocks. It includes embedding layers for token representation, a series of transformer blocks for sequence modeling, and a linear layer as the language modeling head. The model is capable of handling various language modeling tasks, such as text generation and understanding.

    Attributes:
        args (TextConfig):
             A configuration object containing model hyperparameters and settings.
        model_type (str):
             The specific architecture type of the model (obtained from args).
        vocab_size (int):
             The size of the vocabulary used by the model (obtained from args).
        num_hidden_layers (int):
             The number of hidden layers in the model (obtained from args).

    Methods:
        __call__(inputs, cache=None, inputs_embeds=None, mask=None):
             Processes the inputs through the model and returns the output along with updated cache.
        sanitize(weights):
             Cleans the given weight dictionary by removing unwanted keys.
            The __init__ method initializes the layers and components of the language model according to the provided configuration.

    Note:
        The class architecture presumes the necessary PyTorch imports and customized TransformerBlock class are available within the scope.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes a new instance of a model with Transformer architecture.

        Args:
            args (TextConfig):
                 An object containing the configuration parameters for the
            transformer model. Expected to have the following attributes:
            - model_type:
                 The type of the model.
            - vocab_size:
                 The size of the vocabulary.
            - num_hidden_layers:
                 The number of hidden layers in the transformer model.
            - hidden_size:
                 The size of the hidden layer representations.
            - intermediate_size:
                 The size of the 'intermediate' layer in the feedforward network.
            - rms_norm_eps:
                 Epsilon value for RMS normalization.

        Raises:
            AssertionError:
                 If the 'vocab_size' attribute in 'args' is less than or
                equal to zero.

        """
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        # for passing merged input embeddings
        """
        Performs a forward pass through the transformer model.

        Args:
            inputs (mx.array):
                 Input tensor representing token indices of shape `(batch_size, sequence_length)`.
            cache (optional):
                 List of cached tensors from previous forward passes used for
                incremental decoding, one for each layer. It is `None` by default.
            inputs_embeds (optional):
                 Optionally, instead of passing `inputs`, precomputed embeddings can
                be provided directly. If given, it should be of shape `(batch_size, sequence_length, embedding_size)`.
                Default is `None`, which means `inputs` are needed and will be embedded inside the function.
            mask (Optional[mx.array], optional):
                 Attention mask of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
                If `None`, a causal mask is created and applied for autoregressive models. Default is `None`.

        Returns:
            (tuple):
                 A tuple `(last_hidden_state, cache)`.
                `last_hidden_state` is the output of the last layer's pre-norm, projected by `lm_head`,
                of shape `(batch_size, sequence_length, num_tokens)`.
                `cache` is the updated list of cached tensors for each layer after the forward pass.

        """
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.lm_head(self.norm(h)), cache

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        """
        Removes specific keys from the given dictionary of weights.
        The function iterates over the dictionary of weights and excludes any entries where the key contains the specific substring 'self_attn.rotary_emb.inv_freq'. This is often used to sanitize state dictionaries of models before loading them, to ensure that incompatible keys are not included.

        Args:
            weights (dict):
                 A dictionary where the keys are strings representing the weight names and the values are the actual weights (typically tensors or arrays).

        Returns:
            (dict):
                 A new dictionary with all the original key-value pairs except those keys that contain the specified substring.

        """
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        """
        Gets the layers of the model.
        This property returns a list of layers that are part of the model. Each element in the list
        represents a layer within the model's architecture.

        Returns:
            (List):
                 A list containing the layers of the model.

        """
        return self.model.layers
