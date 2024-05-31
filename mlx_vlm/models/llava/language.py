"""

Provides classes and functions to create and work with transformer-based language models.

This module includes the definition of a transformer-based language model architecture called Llama, which stands for Language Models from Large-scale Antiquated Machine Algorithms. The Llama model is designed to handle various NLP tasks by processing sequences of tokens and producing corresponding embeddings or predictions.

Classes:
    - TextConfig: A dataclass that defines the configuration parameters for the text processing models, including the model type, hidden size, number of layers, and vocab size amongst others. It also validates certain configurations post-initialization.
    - Attention: Defines a multi-head attention mechanism with relative positioning encoding (RoPE), used in the transformer blocks.
    - MLP: Implements the feed-forward network used within the transformer blocks, using the gated linear units (SiLU) as its activation function.
    - TransformerBlock: Comprises the attention mechanism and MLP, representing the core computational unit of the transformer.
    - Llama: An implementation of the transformer model configured according to the TextConfig. It sequentially applies TransformerBlock layers and normalizes the output.
    - LanguageModel: A language model wrapper that combines the Llama transformer model with a linear projection head for generating logits over a vocabulary.

Functions:
    LanguageModel.sanitize: A static method to cleanse the model weights, removing unnecessary entries.

Attributes:
    - model.layers: Exposes the transformer layers within the LanguageModel class, allowing external modules to access and, potentially, modify the individual layers.
"""

import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TextConfig:
    """
    A configuration class for text-related model properties.

    Attributes:
        model_type (str):
             The type of text model to configure.
        hidden_size (int):
             Dimensionality of the encoder layers and the pooler layer, defaults to 4096.
        num_hidden_layers (int):
             Number of hidden layers in the encoder, defaults to 32.
        intermediate_size (int):
             The size of the 'intermediate' layer in the transformer encoder, defaults to 11008.
        num_attention_heads (int):
             Number of attention heads for each attention layer in the transformer encoder, defaults to 32.
        rms_norm_eps (float):
             The epsilon used for stability in layer normalization algorithms, defaults to 1e-06.
        vocab_size (int):
             Size of the vocabulary, or number of unique tokens, defaults to 32000.
        num_key_value_heads (int):
             Number of key/value pairs in attention mechanism; if None, set to the same value as num_attention_heads.
        rope_theta (float):
             The theta parameter for RoPE positional encoding, defaults to 10000.
        rope_traditional (bool):
             Flag to indicate using traditional RoPE encoding, defaults to False.
        rope_scaling (Optional[Dict[str, Union[float, str]]]):
             Optional scaling parameters for the RoPE encoding.

    Methods:
        from_dict(cls, params):
             Creates an instance of the class from a dictionary, only including parameters that exist in the class definition.
        Post-initialization:
            On object instantiation, if num_key_value_heads is not explicitly defined, defaults it to num_attention_heads.
            Also, validates the rope_scaling dictionary to contain required keys {'factor', 'type'} and that 'type' key must have value 'linear'. Raises ValueError if validation fails.


    """

    model_type: str
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 32000
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, params):
        """
        Converts a dictionary of parameters into an instance of the cls class.
        The method uses introspection to filter out keys from the input dictionary that are not
        in the signature of the cls class' constructor. The remaining key-value pairs are
        then used to instantiate an object of the cls class.

        Args:
            params (dict):
                 A dictionary of parameters to be used for instantiation.

        Returns:
            An instance of the cls class initialized with the provided parameters that
            match the constructor's signature.

        Raises:
            TypeError:
                 If any of the keys within `params` do not match the constructor's
                signature, or if there are missing required arguments that are not provided
                in `NameError` if cls is not a properly defined class that can be inspected,
                or if `params` is not a dictionary.

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
        Initializes the instance with additional processing after object creation.
        This method runs automatically after the instance is created. It manually sets up
        the number of key value heads if not already specified, and validates the 'rope_scaling'
        attribute ensuring it has the necessary keys and correct type if it is provided.

        Raises:
            ValueError:
                 If 'rope_scaling' is provided but doesn't contain the required keys
                {'factor', 'type'} or if 'rope_scaling' contains a 'type' that isn't 'linear'.

        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")


class Attention(nn.Module):
    """
    Class that implements a scaled dot-product attention mechanism with Rotary Position Embeddings (RoPE).
    This class represents an attention module that computes the attention mechanism over inputs.
    It supports features like multiple attention heads, optional masking, and caching for efficient
    computation. Additionally, it includes Rotary Position Embeddings for capturing positional information.

    Attributes:
        n_heads (int):
             Number of attention heads.
        n_kv_heads (int):
             Number of key/value heads.
        repeats (int):
             The division factor for the number of attention heads by key/value heads.
        scale (float):
             The scaling factor for query normalization.
        q_proj (nn.Linear):
             Linear projection layer for queries.
        k_proj (nn.Linear):
             Linear projection layer for keys.
        v_proj (nn.Linear):
             Linear projection layer for values.
        o_proj (nn.Linear):
             Linear projection layer for the output.
        rope (nn.RoPE):
             Rotary Position Embedding module.

    Args:
        config (TextConfig):
             Configuration object containing model hyperparameters.

    Methods:
        __call__(self, x, mask=None, cache=None):
            Computes the attention mechanism on the input tensor.

    Args:
        x (mx.array):
             Input sequence tensor with shape (Batch, Length, Dimension).
        mask (Optional[mx.array]):
             Optional attention mask with shape (Batch, Length) or (Batch, 1, Length, Length).
        cache (Optional[Tuple[mx.array, mx.array]]):
             Optional tuple of cached key and value projections.

    Returns:
        (Tuple[mx.array, Tuple[mx.array, mx.array]]):
             Tuple containing the attention output and updated cache.


    """

    def __init__(self, config: TextConfig):
        """
        Initializes a new object of the text configuration class.
        This method sets up various components and parameters for attention mechanisms based on the provided configuration. It initializes the number of attention and key-value heads, calculates the dimension of each head, and sets a scaling factor for the query projection. It creates linear projections for query, key, as well as value vectors and a linear output projection. It also initializes a Relative Order Positional Encoding (RoPE) module with configurable scaling and if a linear scaling factor is specified in the configuration, it appropriately adjusts the RoPE scaling.

        Args:
            config (TextConfig):
                 An instance of TextConfig containing the necessary settings to initialize the model components.

        Raises:
            ValueError:
                 If the rope_scaling type in config is neither None nor 'linear', no other scaling types are supported.

        """
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        self.repeats = n_heads // n_kv_heads

        head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = (
            1 / config.rope_scaling["factor"]
            if config.rope_scaling is not None
            and config.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Performs a single call to the attention mechanism with optional caching.
        This method takes an input tensor `x`, an optional mask `mask`, and an optional `cache`. It performs the
        scaled dot product attention mechanism on the input `x` after projecting it into queries, keys,
        and values. The method allows for caching of keys and values to be used for incremental decoding.

        Args:
            x (mx.array):
                 The input tensor of shape (Batch size, Sequence length, Feature dimension).
            mask (Optional[mx.array]):
                 An optional mask tensor for the attention mechanism. The mask specifies which
                positions should not be attended to (masked positions will have a large negative number like -1e9).
            cache (Optional[Tuple[mx.array, mx.array]]):
                 An optional tuple of tensors representing the cached
                keys and values from previous steps. The first tensor in the tuple is the keys cache and
                the second tensor is the values cache.
                This is particularly useful for tasks such as machine translation, where one might want to
                perform decoding incrementally to speed up the process.

        Returns:
            (mx.array):
                 The output tensor after applying attention and a tuple containing updated keys and
                values which can be used for caching.
                The output tensor has the same shape as the input tensor `x`.

        Raises:
            None

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
    A class representing a multi-layer perceptron (MLP) module, extending PyTorch's nn.Module.
    This MLP class implements a simple neural network architecture with a single hidden layer, projecting the input to a
    higher-dimensional space, applying a gated activation function, followed by projecting it back to the original space.

    Attributes:
        gate_proj (nn.Linear):
             A linear layer projecting the input from `dim` dimensions to `hidden_dim` dimensions without bias
        down_proj (nn.Linear):
             A linear layer projecting from `hidden_dim` dimensions back to `dim` dimensions without bias
        up_proj (nn.Linear):
             A linear layer projecting the input from `dim` dimensions to `hidden_dim` dimensions without bias

    Args:
        dim (int):
             The dimensionality of the input and output vectors
        hidden_dim (int):
             The dimensionality of the hidden layer

    Methods:
        __call__(self, x) -> mx.array:
             Applies the MLP operations on the input tensor `x`. The computation involves a gated activation
            which uses the sigmoid-weighted linear unit (SiLU) activation (also known as Swish).
            The method returns the output of the MLP as an MXNet array.

    """

    def __init__(self, dim, hidden_dim):
        """
        Initializes the neural network module with specific projection layers.
        This constructor sets up three linear projection layers within the network module. It inherits from a parent class (not specified),
        and it assumes that the parent class constructor is called with no arguments.

        Args:
            dim (int):
                 The dimensionality of the input features.
            hidden_dim (int):
                 The size of the hidden layer.

        Raises:
            TypeError:
                 If the superclass does not support parameterless initialization.

        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Applies a gating mechanism to the input data followed by a projection.

        Args:
            x (mx.array):
                 The input data on which to perform gating and projection.

        Returns:
            (mx.array):
                 The result of applying the gating mechanism and projection to the input data.
                This method applies a gating mechanism to the input array x, then uses a
                sigmoid-weighted linear unit (SiLU) as the activation function. The result of
                this activation is element-wise multiplied by the result of an upward projection, and the outcome is then passed through a downward projection to produce the final output.

        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    Class TransformerBlock is a PyTorch nn.Module representing a single block of a Transformer model.
    This block includes components such as multi-head self-attention, position-wise feed-forward networks (MLP),
    and layer normalization. The configuration for the block is provided through the `TextConfig` class,which
    should include the number of attention heads, hidden size, intermediate size for the MLP, and epsilon for
    RMS normalization.

    Attributes:
        num_attention_heads (int):
             The number of attention heads in the multi-head attention mechanism.
        hidden_size (int):
             The hidden size of the attention representations.
        self_attn (Attention):
             The multi-head self-attention mechanism.
        mlp (MLP):
             The multi-layer perceptron (feed-forward network) following self-attention.
        input_layernorm (nn.RMSNorm):
             Layer normalization applied before self-attention.
        post_attention_layernorm (nn.RMSNorm):
             Layer normalization applied after self-attention.
        config (TextConfig):
             The configuration object containing model hyperparameters.

    Methods:
        __call__(self, x, mask=None, cache=None):
            Defines the computation performed at every call of the TransformerBlock.

    Args:
        x (mx.array):
             The input tensor to the Transformer block.
        mask (Optional[mx.array], optional):
             The mask tensor for the self-attention mechanism.
            Defaults to None.
        cache (Optional[Tuple[mx.array, mx.array]], optional):
             Cached intermediate results to
            facilitate incremental decoding. Defaults to None.

    Returns:
        (Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]):
             A tuple containing the output tensor after
            processing through the Transformer block and the updated cache, if provided in input.

    """

    def __init__(self, config: TextConfig):
        """
        Initializes an instance of the model with specified configurations.
        This initialization method sets up the model's core components including attention
        heads, hidden size parameters, and the multilayer perceptron (MLP) according to
        the provided TextConfig object. Additionally, it configures layer normalization
        using root mean square normalization with the specified epsilon value from the
        configuration.

        Args:
            config (TextConfig):
                 A configuration object containing parameters for
                model architecture such as the number of attention heads, hidden layer
                size, intermediate size for the MLP, and normalization epsilon.

        Raises:
            ValueError:
                 If the configuration parameters do not meet the requirements
                for initializing the layers, such as incompatible sizes or mismatches
                in dimensions.

        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Calls the object as a function and processes the input through the attention mechanism.

        Args:
            x (mx.array):
                 The input tensor that will be passed through the model.
            mask (Optional[mx.array], optional):
                 The optional mask tensor that can be used to mask certain elements
                from the input tensor during the attention process. Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]], optional):
                 Optional tuple of tensors representing
                cached intermediate results from previous steps. Can be used to improve performance on sequential
                inputs by avoiding redundant computations. Defaults to None.

        Returns:
            (mx.array):
                 A tensor that represents the output of the function after applying self-attention
                and feed-forward layers, along with the updated cache.

        Raises:
            TypeError:
                 If the input types are not as expected.

        """
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class Llama(nn.Module):
    """
    A Llama class inheriting from PyTorch's nn.Module that represents a language model for text generation or understanding tasks.
    The class encapsulates a Transformer-like architecture with a specified configuration. It includes an embedding
    layer for input tokens and a number of transformer blocks as specified in the configuration. Additionally, it
    applies normalization to the output of the last transformer layer using Root Mean Square Layer Normalization (RMSNorm).

    Attributes:
        config (TextConfig):
             Configuration object containing parameters like vocab_size, hidden_size, and
            num_hidden_layers, among others necessary for initializing the Llama model components.
        vocab_size (int):
             The total number of unique tokens in the vocabulary.
        num_hidden_layers (int):
             The number of hidden transformer layers in the model.
        embed_tokens (nn.Embedding):
             Embedding layer to convert input token indices into embeddings.
        layers (List[TransformerBlock]):
             A list of transformer blocks that define the model's core architecture.
        norm (nn.RMSNorm):
             Layer normalization module applied to the final outputs of the transformer layers.

    Methods:
        __call__(inputs, cache=None, inputs_embeds=None):
            Forward pass through the Llama model. It allows optional caching for efficient inference and has the ability to
            accept pre-computed embeddings.

    Args:
        inputs (mx.array):
             An array of input token indices.
        cache (optional):
             A list of cached outputs from previous forward passes. Defaults to None.
        inputs_embeds (optional):
             Precomputed token embeddings. If provided, these will be used instead of the
            input token indices. Defaults to None.

    Returns:
        Tuple of the normalized output after all transformer layers (mx.array), and the updated cache (List).

    Raises:
        AssertionError:
             If the 'vocab_size' attribute extracted from the given 'config' is not greater than 0.

    """

    def __init__(self, config: TextConfig):
        """
        Initializes a new instance of a neural network model with the provided `TextConfig` configuration.

        Args:
            config (TextConfig):
                 A configuration object containing model hyperparameters such as vocabulary size, number of hidden layers, hidden size, and normalization epsilon values.

        Raises:
            AssertionError:
                 If the `vocab_size` specified in the configuration object is not greater than zero.

        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
    ):
        # for passing merged input embeddings
        """
        Performs a forward pass through the model's Transformer layers.
        This method processes input tensor embeddings and performs a sequence of transformations through the Transformer's layers. If embedding representations are not provided, they are created from the input tokens using the embedding layer. A causal mask is created and applied during attention operations to prevent tokens from attending to subsequent tokens, maintaining the autoregressive property of the model. Each Transformer layer updates the embeddings and returns new hidden states along with updated cache states for further reuse in incremental decoding tasks.

        Args:
            inputs (mx.ndarray):
                 A tensor of input token IDs of shape (batch_size, sequence_length).
            cache (Optional[List[Any]]):
                 A list of layer cache states from a previous forward pass, used for faster subsequent predictions in sequence generation tasks. Defaults to None.
            inputs_embeds (Optional[mx.ndarray]):
                 Precomputed token embeddings, providing an alternative to computing embeddings from `inputs`. If None, embeddings are computed from `inputs`. Defaults to None.

        Returns:
            (Tuple[mx.ndarray, List[Any]]):
                 A tuple containing the normalized final hidden states of shape (batch_size, sequence_length, hidden_size) and an updated list of cache states for each layer.

        Raises:
            AssertionError:
                 If there are inconsistencies detected in input shapes or mask operations.

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

        return self.norm(h), cache


class LanguageModel(nn.Module):
    """
    A neural network module designed for language tasks, specifically with the 'llama' model type.
    This class is a subclass of `nn.Module` and encapsulates a language model based on the parameters specified
    in the provided `TextConfig` object. The model consists of a 'llama' as the underlying language model component,
    along with a linear transformation used as a language model head.

    Attributes:
        model_type (str):
             Indicates the type of model based on the configuration (`'llama'` is expected).
        model:
             The `Llama` language model instance initialized with the given config.
        lm_head:
             A linear layer that projects the output of the `Llama` model to the vocabulary size.

    Raises:
        ValueError:
             If the `model_type` in the provided config is not `'llama'`, an exception is raised.

    Methods:
        __init__(config:
             TextConfig):
            Initializes the language model with the given configuration.
        __call__(inputs:
             mx.array, cache=None, inputs_embeds=None, mask: Optional[mx.array]=None):
            Processes input data through the model and returns the output logits along with the updated cache if applicable.
        sanitize(weights):
            Removes certain unwanted weight parameters from the provided state dictionary.
        layers:
            A property that returns the layers from the underlying `Llama` model.

    """

    def __init__(self, config: TextConfig):
        """
        Initializes a text generating model with a specific configuration.

        Args:
            config (TextConfig):
                 The configuration object containing model settings such as model type, vocabulary size,
                number of hidden layers, hidden size, and any other relevant parameters required to define the model structure.

        Raises:
            ValueError:
                 If the specified model type in the configuration is not 'llama', since only the 'llama' model type is
                supported currently.

        Attributes:
            model_type (str):
                 The type of model to be initialized based on the configuration, typically expected to be 'llama'.
            model (Llama):
                 The Llama model instance created using the provided configuration.
            lm_head (nn.Linear):
                 The linear layer that projects the hidden states to the vocabulary space, used in language modeling.

        """
        super().__init__()
        self.model_type = config.model_type
        if self.model_type != "llama":
            raise ValueError(
                f"Model type {self.model_type} not supported. Currently only 'llama' is supported"
            )
        self.model = Llama(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        """
        Performs a forward pass of the model with the given inputs.

        Args:
            inputs (mx.array):
                 Input tensor to the model.
            cache (optional):
                 The cache used for stateful/persistent computation across calls.
            inputs_embeds (optional):
                 Precomputed embeddings for the inputs.
            mask (Optional[mx.array]):
                 The mask tensor specifying which elements to mask during the computation.

        Returns:
            (tuple):
                 A tuple containing the output logits after applying the language model head, and the updated cache.

        """
        out, cache = self.model(inputs, cache, inputs_embeds)
        return self.lm_head(out), cache

    @staticmethod
    def sanitize(weights):
        # Remove unused precomputed rotary freqs
        """
        Sanitizes a dictionary of weights by removing any entries which are related to self attention rotary embeddings.
        The function filters out specific keys within the input dictionary that contain the substring
        'self_attn.rotary_emb.inv_freq', as these keys are related to rotary embeddings which may need
        special handling or exclusion from certain operations.

        Args:
            weights (dict):
                 A dictionary where keys are the names of weights and values are the actual
                numerical weights, typically tensor-like objects.

        Returns:
            (dict):
                 A new dictionary with the specified entries removed.

        """
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        """
        Gets the layers of the associated model.
        This property provides access to the list of layers that constitute the model. It is a
        read-only property that returns the layers upon being called.

        Returns:
            (list):
                 A list containing the layers of the model.

        """
        return self.model.layers
