"""

The language module is a comprehensive collection of classes for constructing and using a transformer-based neural network architecture specifically tailored for natural language processing tasks. At the core of this module is a class `LanguageModel` that leverages a `Qwen2Model`, which in itself is a sequence of `TransformerBlock` modules. Each `TransformerBlock` consists of attention mechanisms and multi-layer perceptrons (MLPs) to process input sequences for tasks such as language modeling or text generation.

Classes:
- `TextConfig`: A data class that stores the configuration parameters for the language model. It supports creating model configurations from a dictionary and validates the necessary parameters post-initialization.

- `Attention`: Implements a multi-head attention mechanism, including RoPE (Rotary Position Embedding) for relative positional encoding.

- `MLP`: Represents a multi-layer perceptron with gated linear units and projection layers for modeling complex transformations within the neural network.

- `TransformerBlock`: Encapsulates a single transformer layer that includes self-attention and an MLP, along with normalization layers before and after the self-attention.

- `Qwen2 [Back-Quotes]Model`: Constructs a model with multiple layers of `TransformerBlock` and handles input token embeddings, cache for transformer states, and output projections to vocabulary size.

- `LanguageModel`: Creates an instance of `Qwen2Model` and provides an interface to interact with the underlying transformer model, including the ability to update the weights and extract individual layers.

All classes are designed with modularity and flexibility in mind, ensuring the ability to adapt to different configurations and requirements of various NLP applications. The module provides the necessary tools and abstractions for training and fine-tuning language models or for integrating pre-trained models into downstream tasks.
"""

import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TextConfig:
    """
    A dataclass representing configuration parameters for a text model.

    Attributes:
        model_type (str):
             Type of the model.
        hidden_size (int):
             Size of the hidden layers.
        num_hidden_layers (int):
             Number of hidden layers.
        intermediate_size (int):
             Size of the 'intermediate' (i.e., feed-forward) layer.
        num_attention_heads (int):
             Number of attention heads.
        rms_norm_eps (float):
             Epsilon parameter for RMS normalization.
        vocab_size (int):
             Size of the model's vocabulary.
        num_key_value_heads (int):
             Number of key/value pairs in attention mechanism, defaults to `num_attention_heads` if not provided.
        rope_theta (float):
             Theta parameter for RoPE encoding. Defaults to 1000000.
        rope_traditional (bool):
             A flag indicating whether traditional RoPE encoding is used. Defaults to False.
        rope_scaling (Optional[Dict[str, Union[float, str]]]):
             Parameters for scaling the RoPE encoding. Defaults to None.
        tie_word_embeddings (bool):
             A flag indicating whether to tie input and output word embeddings. Defaults to True.

    Methods:
        from_dict(cls, params):
             Class method that creates an instance of `TextConfig` from a dictionary, ensuring only relevant parameters are passed.
        __post_init__(self):
             A post-initialization method to set default values and validate parameters.

    Raises:
        ValueError:
             If `rope_scaling` is provided but does not contain the required keys or if 'type' in `rope_scaling` is not set to 'linear'.

    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int = None
    rope_theta: float = 1000000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params):
        """
        Creates an instance of the class using the given dictionary `params`.
        This class method takes a dictionary `params`, filters its items based on the class constructor's signature,
        and then unpacks the filtered dictionary to the class constructor to create an instance of the class.

        Parameters:
            -----------
            params :
                 dict
                A dictionary where keys are string representations of the class constructor parameters, and values
                are the corresponding values to be passed to the class constructor.

        Returns:
            -------
            object
            An instantiated object of the class.

        Raises:
            ------
            TypeError
            If any keys in `params` are not found in the class constructor's signature.

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
        Initializes additional properties of the instance after the main __init__ has run.

        Raises:
            ValueError:
                 If the 'num_key_value_heads' attribute is not set, it defaults to the value of
                'num_attention_heads'. If the 'rope_scaling' attribute is provided, it
                must contain the keys 'factor' and 'type'. Also, 'rope_scaling' only
                supports the 'linear' type, and raises an error if a different type is
                specified.

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
    Attention module applying scaled dot product attention mechanism.
    This class implements the attention mechanism which is a critical component in
    transformer-based models. It computes the attention weights and applies them to the
    values using a scaled dot product approach.

    Parameters:
        args (TextConfig):
             Configuration object containing model hyperparameters.

    Attributes:
        n_heads (int):
             Number of attention heads.
        n_kv_heads (int):
             Number of key/value attention heads.
        scale (float):
             Scaling factor for the dot products.
        q_proj (nn.Linear):
             Linear projection layer for queries.
        k_proj (nn.Linear):
             Linear projection layer for keys.
        v_proj (nn.Linear):
             Linear projection layer for values.
        o_proj (nn.Linear):
             Linear projection layer for the output.
        rope (nn.RoPE):
             Relative positional encoding module.

    Methods:
        __call__(x, mask=None, cache=None):
            Computes the attention mechanism on input tensor x.

    Args:
        x (mx.array):
             Input tensor of shape (Batch, Seq_len, Dim).
        mask (Optional[mx.array]):
             Optional mask for the attention weights.
        cache (Optional[Tuple[mx.array, mx.array]]):
             Tuple containing previous keys and values
            for incremental decoding (used mainly in language modeling).

    Returns:
        (mx.array):
             The result of the attention computation.
        (Tuple[mx.array, mx.array]):
             Tuple containing updated keys and values, serving
            as cache for subsequent incremental decoding steps.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes the Attention module with the provided configuration.

        Args:
            args (TextConfig):
                 A configuration object containing the necessary parameters for the attention module.
            The expected attributes within args are:
            - hidden_size (int):
                 The size of the hidden layer.
            - num_attention_heads (int):
                 The number of attention heads.
            - num_key_value_heads (int):
                 The number of key-value heads.
            - rope_scaling (dict, optional):
                 A dictionary containing the rope scaling factor.
            - rope_traditional (bool, optional):
                 A flag to use traditional relative positional encoding.
            - rope_theta (float, optional):
                 The base theta value for RoPE.

        Raises:
            Attribute Error:
                 If any required attribute is missing in the args.

        """
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Performs a forward pass through a self-attention mechanism with optional caching.
        This method computes the attention weights and output values for given input, projection matrices,
        and optional cached keys and values. It uses scaled dot product attention and can also apply
        relative position encoding via the rope method. The output is a tuple containing the projected
        output and updated caching tensors if caching was used.

        Args:
            x (mx.array):
                 The input tensor with shape (batch_size, seq_length, model_dim).
            mask (Optional[mx.array], optional):
                 An optional mask to exclude certain positions from
                the attention mechanism. Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]], optional):
                 Optional cached tensors of
                keys and values from previous attention calculations to be used for incremental
                decoding. Defaults to None.

        Returns:
            (Tuple[mx.array, Tuple[mx.array, mx.array]]):
                 A tuple, where the first element is the output
                tensor of the self-attention with shape (batch_size, seq_length, model_dim) and the second
                element is a tuple of cached keys and values.

        Raises:
            mx.MXNetError:
                 If there is an issue with the attention operation using MXNet internals.

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
    A Multilayer Perceptron (MLP) module that implements a simple feed-forward neural network architecture with gating mechanisms.
    This class inherits from PyTorch's `nn.Module` and represents a neural network with one hidden layer that includes a gating mechanism. The gating mechanism is achieved by element-wise multiplication of a gated projection and an up projection of the input tensor.

    Attributes:
        gate_proj (nn.Linear):
             Linear transformation without bias from input dimension to hidden dimension.
        down_proj (nn.Linear):
             Linear transformation without bias from hidden dimension to input dimension, acting as a gating mechanism.
        up_proj (nn.Linear):
             Linear transformation without bias from input dimension to hidden dimension, typically increasing representational capacity.

    Args:
        dim (int):
             The size of the input dimension for the linear transformations.
        hidden_dim (int):
             The size of the hidden layer dimension that expands or contracts the input feature representation.

    Returns:
        (mx.array):
             The output of the MLP after processing the input `x` is returned as an MXNet array.

    Note:
        nn.silu is used as a non-linear activation function between the gate and down projections.


    """

    def __init__(self, dim, hidden_dim):
        """
        Initializes a new instance of the class with the specified dimensions for projections.
        This constructor method sets up three linear projection layers within a module. It defines a 'gate' projection, a 'down' projection,
        and an 'up' projection. Each projection maps to a different size, either increasing or decreasing the dimensionality
        as specified by the parameters. This method does not apply bias to the 'gate' and 'down' projection layers.

        Args:
            dim (int):
                 The dimensionality of the input features.
            hidden_dim (int):
                 The dimensionality of the hidden layer features.

        Raises:
            TypeError:
                 If `dim` or `hidden_dim` are not of type int.


        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Performs a gated transformation on the input using a sequence of operations.
        This method applies a gating mechanism over the input data `x`. It carries out a downscaling projection, followed by a
        scaled exponential linear unit (SiLU) activation function, and an element-wise multiplication with an
        upscaling projection of `x`. The result is a transformed `mx.array` that has undergone a gated operation, which
        can be beneficial in regulating and learning complex representations in neural network layers.

        Args:
            awrgs:
            x:
                 An `mx.array` input tensor to be processed.

        Returns:
            An `mx.array` after applying the gated transformation.

        Raises:
            No explicit raises but depends on the underlying neural network layer's exceptions.

        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    A TransformerBlock module as part of a neural network architecture, which encapsulates the mechanisms of
    multi-headed self-attention and a subsequent feed-forward neural network (MLP), including layer normalization
    steps before and after the self-attention operation.
    This class inherits from `nn.Module` and utilizes a configuration object `args` of type `TextConfig` to
    initialize various components of the transformer block. The `TextConfig` object holds parameters such
    as the number of attention heads (`num_attention_heads`), the hidden layer size (`hidden_size`), intermediate size
    for the MLP (`intermediate_size`), and the epsilon value for RMS normalization (`rms_norm_eps`).

    Attributes:
        num_attention_heads (int):
             The number of attention heads in the multi-head attention mechanism.
        hidden_size (int):
             The size of the hidden layer representations.
        self_attn (Attention):
             The self-attention mechanism for the transformer block.
        mlp (MLP):
             The multilayer perceptron that follows the self-attention mechanism.
        input_layernorm (nn.RMSNorm):
             Layer normalization applied before the self-attention mechanism.
        post_attention_layernorm (nn.RMSNorm):
             Layer normalization applied after the self-attention mechanism.
        args (TextConfig):
             The configuration object containing all necessary parameters for the transformer block.
            The module's `__call__` method allows the transformer block to be called with an input tensor `x`, an optional
            mask `mask`, and an optional cache `cache` for use in inference for sequence generation tasks. The method
        performs the following operations in sequence:
             layer normalization on the input, self-attention with potential
            caching, residual connection, another layer normalization, feed-forward network application, and a second
            residual connection to produce the final output of the block.
            The module returns a tuple `(out, cache)`, where `out` is the transformed input and `cache` holds the current
            state of the attention heads for use in subsequent calls (e.g., in a decoding loop).

    """

    def __init__(self, args: TextConfig):
        """
        Initializes the transformer component with attention and MLP layers.

        Args:
            args (TextConfig):
                 Configuration object that contains parameters such as the number of
                attention heads, hidden size, intermediate size, and RMS normalization epsilon.
                It provides all the necessary information to properly configure the transformer component.


        Raises:
            ValueError:
                 If the configuration parameters provided in `args` are invalid,
                the initialization may raise ValueError exceptions from the Attention or MLP
                class constructors.

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
        Calls the layer on a given input and returns the output with the updated cache.
        This method applies a self-attention mechanism to the input followed by a multi-layer
        perceptron (MLP) to compute the layer's output. The self-attention takes into account an optional
        mask and makes use of a cache for more efficient computation. Layer normalization is applied both
        before the self-attention and after the self-attention but before the MLP.

        Args:
            x (mx.array):
                 The input array to the layer.
            mask (Optional[mx.array]):
                 An optional mask to be applied during self-attention. Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]]):
                 An optional tuple containing cached arrays for
                more efficient computation in sequential processing. Defaults to None.

        Returns:
            (mx.array):
                 The output array of the layer after the self-attention and MLP operations.

        Raises:
            TypeError:
                 If the input types are not as expected.
            ValueError:
                 If the input values are not within expected ranges or if they're invalid in the current context.

        """
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class Qwen2Model(nn.Module):
    """
    A PyTorch module for a token-level transformer model with causal attention.
    The Qwen2Model is designed to be a transformer-based neural network
    for language modeling tasks. It takes an input sequence of tokens and predicts
    the next token in the sequence. This model is composed of multiple transformer
    blocks, each employing multi-head self-attention, and is capable of handling
    input with an added causal mask for sequence generation tasks.

    Attributes:
        args (TextConfig):
             Configuration object containing model hyperparameters.
        vocab_size (int):
             Size of the token vocabulary.
        num_hidden_layers (int):
             Number of hidden layers in the transformer model.
        embed_tokens (nn.Embedding):
             Token embedding layer.
        layers (List[TransformerBlock]):
             List of transformer blocks constituting the
            core of the model.
        norm (nn.RMSNorm):
             Normalization layer applied before the final linear layer.
        lm_head (nn.Linear):
             Final linear layer that projects hidden states to logits
            over the vocabulary.

    Raises:
        AssertionError:
             If the `vocab_size` in `args` is less than or equal to zero.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes a new instance of a model with transformer blocks.
        This constructor method creates a new model with an embedding layer for token lookup,
        several transformer blocks for encoding text into a meaningful representation, and
        a linear projection layer as the language modeling head. The parameters for configuring
        the model are provided through the instance of Text JamesConfig class.

        Args:
            args (TextConfig):
                 The configuration object that contains model-specific
                parameters such as the number of hidden layers, the size of the vocabulary,
                and the hidden size of the embeddings.

        Raises:
            AssertionError:
                 If the vocabulary size specified in args is not greater than 0.


        """
        super().__init__()
        self.args = args
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
    ):
        # for passing merged input embeddings
        """
        Calls the transformer model on input data, optionally using cached states and input embeddings.
        This method processes inputs through token embedding (if inputs_embeds is None), generates a
        causal attention mask for preventing future information leakage, and passes the data through
        multiple transformer layers. The output of the transformer layers is normalized and then passed
        to a linear layer to generate the final output logits. The cache is updated with the current layer's
        state for each layer in the transformer.

        Args:
            inputs (mx.array):
                 Token indices for the input sequence.
            cache (Optional[None, List[Any]]):
                 Cached layer states from previous forward calls, utilized for
                incremental decoding. Defaults to None, indicating no caching.
            inputs_embeds (Optional[None, Any]):
                 Precomputed token embeddings for the input sequence.
                Defaults to None, indicating the embeddings will be computed from `inputs`.

        Returns:
            (Tuple[Any, List[Any]]):
                 A tuple containing the output logits from the linear layer after the
                transformer layers (as applied to the normalized hidden states), and the updated cache with
                the states from the current forward pass.

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


class LanguageModel(nn.Module):
    """
    A PyTorch module that represents a language model with configurable parameters and model type.

    Attributes:
        args (TextConfig):
             A configuration object containing model parameters.
        model_type (str):
             A string that defines the type of model derived from args.
        model (Qwen2Model):
             An instance of Qwen2Model initialized with args.

    Methods:
        __init__(self, args:
             TextConfig):
            Initializes the LanguageModel instance.

    Args:
        args (TextConfig):
             A configuration object containing the arguments needed for model initialization.
        __call__(self, inputs:
             mx.array, cache=None, inputs_embeds=None, mask: Optional[mx.array]=None):
            Enables the object to be called like a function, processing the inputs through the model.

    Args:
        inputs (mx.array):
             Input tensor to the language model.
        cache:
             Optional; The cache to be used by the model.
        inputs_embeds:
             Optional; Alternative to inputs, precomputed embedding tensor.
        mask (Optional[mx.array]):
             Optional; Mask tensor for the inputs.

    Returns:
        A tuple containing the output tensor and updated cache.
        sanitize(self, weights):
            Sanitizes and filters the incoming weights when loading pre-trained models.

    Args:
        weights (dict):
             A dictionary of the model's weights.

    Returns:
        A dictionary of sanitized weights.
        layers (property):
            Exposes the layers of the internal Qwen2Model instance as a property.

    Returns:
        The model layers.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes a new instance of a model configuration class.

        Args:
            args (TextConfig):
                 An instance of TextConfig class which contains model configuration parameters such as model_type, vocab_size, num_hidden_layers, hidden_size, rms_norm_eps, and any other necessary model parameters.

        Raises:
            AssertionError:
                 If the `vocab_size` attribute of the `args` parameter is not greater than 0, indicating an invalid configuration.

        """
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        """
        Performs a forward pass of the model using the provided inputs and optional cache and input embeddings, with an optional input mask.

        Args:
            inputs (mx.array):
                 The input data array to the model.
            cache (optional):
                 The optional cache from previous states to be used for incremental decoding. Defaults to None.
            inputs_embeds (optional):
                 Pre-computed input embeddings which can be used instead of the model's internal embedding computation. Defaults to None.
            mask (Optional[mx.array], optional):
                 An optional mask that can be applied to the inputs. It is used to hide certain elements of the input from the model. Defaults to None.

        Returns:
            (Tuple):
                 A tuple containing the output of the model and the updated cache.

        Raises:
            ValueError:
                 If unsupported inputs, types, or configurations are provided.

        """
        out, cache = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        return out, cache

    def sanitize(self, weights):
        """
        Sanitizes the model weights by ensuring the tying of word embeddings and removing specific keys.
        This function is responsible for cleaning up the provided weights dictionary. It ensures that if the model arguments
        specify the tying of word embeddings, the appropriate weights are copied from the embedding tokens to the
        language model head. Furthermore, it filters out entries that are not required by removing those with keys
        matching 'self_attn.rotary_emb.inv_freq'. The resulting dictionary with sanitized weights is returned.

        Args:
            weights (dict):
                 A dictionary where keys are parameter names and values are torch.Tensor objects representing
                the parameters of a pre-trained model.

        Returns:
            (dict):
                 A sanitized version of the weights dictionary, where unwanted keys are removed and word embeddings
                are potentially tied, according to the model's configuration.

        Raises:
            ValueError:
                 If 'language_model.model.lm_head.weight' is expected to be in the weights but is missing,
                signaling an inconsistency in the expected model architecture.

        """
        if (
            self.args.tie_word_embeddings
            and "language_model.model.lm_head.weight" not in weights
        ):
            weights["language_model.model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]

        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        """
        Gets the layers of the model.
        This property provides access to the list of layers that make up the model.

        Returns:
            (List[Layer]):
                 A list of layers contained within the model.

        """
        return self.model.layers
