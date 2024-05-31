"""

The language module provides a robust framework for building transformer-based natural language processing (NLP) models. It centers around the `LanguageModel` class, which is designed to handle language tasks by encapsulating a transformer model within it. This module also includes several supporting classes and components that comprise the transformer model, such as `TextConfig`, `RMSNorm`, `Attention`, `MLP`, `TransformerBlock`, and `GemmaModel`. These components are pieced together to create a complete and customizable NLP model that can process text inputs and generate language-based outputs. The module uses a mix of standard neural network layers and custom-designed components tailored specifically for processing language data.

The `TextConfig` dataclass provides a configuration schema for the model, containing necessary parameters like model type, size, layer counts, and other hyperparameters relevant to the transformer model's architecture.

The `RMSNorm` class is a layer for root mean square layer normalization, which is a normalization technique that works well with transformers.

The `Attention` class implements multi-head attention with an optional relative position encoding mechanism through the rotary position encoding (RoPE) system, enabling the model to infer the order of the sequence.

The `MLP` class represents a multi-layer perceptron used within the transformer blocks for additional nonlinearity and feature extraction.

The `TransformerBlock` class encapsulates a single block of the transformer, consisting of an attention mechanism, normalization layers, and an MLP, representing the main building block of the transformer architecture.

The `GemmaModel` class assembles multiple `TransformerBlock` instances to form a deep transformer model, which can process sequences of tokens and capture the complex relationships between them.

Lastly, the `LanguageAdaptiveModel` serves as the top-level API for building language models, with methods to interact with the underlying `GemmaModel` and handle input token sequences, manage caching for efficient incremental token processing, and convert the transformer's output into token embeddings.

Note: The module is designed with a focus on customization and efficiency, utilizing both standard and proprietary components of the assumed `mlx` library. Care has been taken to lay out a scalable and modular architecture, which can be adjusted for various model sizes and complexities, according to the user's requirements.
"""

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TextConfig:
    """
    A class to encapsulate the configuration parameters for text models.

    Attributes:
        model_type (str):
             The type of the model to be configured.
        hidden_size (int):
             The size of the hidden layers in the model.
        num_hidden_layers (int):
             The total number of hidden layers in the model.
        intermediate_size (int):
             The size of the 'intermediate' layer in the model.
        num_attention_heads (int):
             The number of attention heads in the model.
        num_key_value_heads (int):
             The number of key/value heads in the model, typically for attention mechanisms.
        vocab_size (int):
             The size of the vocabulary supported by the model.
        rms_norm_eps (float, optional):
             The epsilon value to use for RMS normalization. Defaults to 1e-06.
        rope_theta (float, optional):
             The theta hyperparameter used for 'Rotary Positional Embeddings' (RoPE). Defaults to 10000.
        rope_traditional (bool, optional):
             A flag to indicate whether to use traditional positional encoding rather than RoPE. Defaults to False.

    Methods:
        from_dict(params):
             Class method to create a TextConfig object from a dictionary, where only keys that correspond to the class attributes are considered for instantiation.


    """

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000
    rope_traditional: bool = False

    @classmethod
    def from_dict(cls, params):
        """
        Constructs an instance of the class from a dictionary of parameters.
        This method is designed to initialize an instance of the class with attributes
        matching the keys in the input dictionary. It filters the dictionary to remove
        any keys that do not correspond to the class's constructor parameters. It then
        uses the filtered parameters to create a new instance of the class using the
        ** unpacking operator.

        Args:
            params (dict):
                 A dictionary where keys are string representations of the
                class constructor's parameter names, and values are the corresponding
                values to be set for those parameters.

        Returns:
            An instance of the class, initialized with the parameters provided in the
            input dictionary that match the constructor's signature.

        Raises:
            TypeError:
                 If any of the keys in the input dictionary do not correspond
                to the constructor's parameter names.

        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class RMSNorm(nn.Module):
    """
    A normalization layer that performs Root Mean Square Layer Normalization (RMSNorm) on inputs.
    This layer is a subclass of `nn.Module` and is intended to be used in neural networks to stabilize the
    learning process by normalizing the input values. RMSNorm normalizes the input tensor across its last dimension
    (the 'features' dimension) by using the Root Mean Square value.

    Attributes:
        weight (mx.ndarray):
             A learnable weight parameter initialized as ones, with shape specified by the `dims` input.
        eps (float):
             A small constant added to the denominator for numerical stability during normalization.
        dims (int):
             The dimension of the input features that require normalization.

    Args:
        dims (int):
             The number of dimensions of the input features that the layer will normalize.
        eps (float, optional):
             A small increment to ensure numerical stability during normalization. Defaults to 1e-06.

    """

    def __init__(self, dims: int, eps: float = 1e-6):
        """
        Initializes a new instance of the given class.
        This constructor initializes an instance with the given dimensions for the weights
        and a small epsilon value for numerical stability during operations.

        Args:
            dims (int):
                 The number of dimensions for the weights.
            eps (float, optional):
                 A small constant to avoid division by zero or other
                numerical issues. Defaults to 1e-06.

        """
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        """
        Performs normalization on the input using root mean square (RMS) with an added learnable weight.

        Args:
            x (Tensor):
                 The input tensor that needs to be normalized.

        Returns:
            (Tensor):
                 The normalized tensor after applying RMS normalization with an additional weight.

        """
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    """
    A PyTorch module that implements a scaled dot-product attention mechanism with optional Rotary Position Embedding (RoPE).
    The Attention module uses a query, key, and value projection mechanism to perform attention over input tensors.
    It supports optional caching of key and value tensors for incremental state updates which is useful
    in tasks such as autoregressive generation.

    Attributes:
        n_heads:
             An integer representing the number of attention heads.
        n_kv_heads:
             An integer representing the number of key/value attention heads.
        scale:
             A scaling factor for the attention scores, calculated as the inverse square root of the head dimension.
        q_proj:
             A Linear layer for projecting input tensor to query tensor.
        k_proj:
             A Linear all-to-all layer for projecting input tensor to key tensor.
        v_proj:
             A Linear layer for projecting input tensor to value tensor.
        o_proj:
             A Linear layer for projecting concatenated outputs to the original tensor dimensions.
        rope:
             An instance of the RoPE class for applying Rotary Position Embedding if desired.

    Args:
        args (TextConfig):
             Configuration containing model hyperparameters like the number of attention heads,
            hidden size, etc.

    Note:
        The input TextConfig class is assumed to have the following attributes:
        - hidden_size:
             The size of the hidden layer in the model.
        - num_attention_heads:
             The number of attention heads for the projection.
        - num_key_value_heads:
             The number of key/value heads for the projection.
        - rope_traditional:
             A boolean indicating if traditional RoPE is used.
        - rope_theta:
             The theta value for the RoPE embedding.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes the component with the specific configuration parameters for attention mechanisms.

        Args:
            args (TextConfig):
                 A configuration object containing various hyperparameters.
                `hidden_size` is the dimensionality of the input and output features.
                `num_attention_heads` specifies the number of attention heads.
                `num_key_value_heads` is the number of heads for key/value projections.
                `rope_traditional` indicates if traditional relative position encoding is used.
                `rope_theta` is a hyperparameter for the relative position encoding.

        Attributes:
            n_heads (int):
                 The number of attention heads.
            n_kv_heads (int):
                 The number of key/value projection heads.
            scale (float):
                 The scaling factor for the query vectors.
            q_proj (nn.Linear):
                 The linear projection layer for queries.
            k_proj (nn.Linear):
                 The linear projection layer for keys.
            v_proj (nn.Linear):
                 The linear projection layer for values.
            o_proj (nn.Linear):
                 The output linear projection layer.
            rope (nn.RoPE):
                 The relative position encoding layer.

        Raises:
            ValueError:
                 If `hidden_size` is not a multiple of the number of attention heads, a ValueError is raised to indicate the mismatch.

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
        Computes the output of a multi-head self-attention layer with optional caching for transformer models.

        Args:
            x (mx.array):
                 The input tensor with shape (Batch, Length, Dimension).
            mask (Optional[mx.array]):
                 An optional mask tensor to be applied to the attention scores before
                softmax. Default is None.
            cache (Optional[Tuple[mx.array, mx.array]]):
                 Optional tuple containing cached keys and
                values tensors. This is used for efficient decoding and is typically used in
                transformer models during inference. Default is None.

        Returns:
            (Tuple[mx.array, Tuple[mx.array, mx.array]]):
                 A tuple containing the projected output tensor
                after applying attention and a tuple with the updated keys and values tensors that
                can be used for caching in sequential decoding steps.

        Raises:
            ValueError:
                 If 'queries', 'keys', or 'values' projections result in mismatched
                dimensions when attempting to reshape or transpose the tensors for multi-head
                attention calculations.

        Note:
            The method assumes that 'queries', 'keys', and 'values' are projections of the input
            'x'. It involves reshaping and transposing these tensors to align with the number of
            attention heads specified in the model configuration.

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
    A multi-layer perceptron (MLP) module that encapsulates a simple feedforward neural network architecture with one hidden layer.
    The MLP class is derived from 'nn.Module', which is a base class for all neural network modules in PyTorch. It is designed to take inputs of a certain dimensionality ('dim') and project them into a hidden layer of a different dimensionality ('hidden_dim') before mapping them back to the original dimensionality.
    It uses a gating mechanism where the input is first transformed by a 'gate projection' linear transformation without a bias, then processed with a GELU (Gaussian Error Linear Unit) activation function, and finally element-wise multiplied by the output of an 'up projection' linear transformation without a bias. The result of this product is then projected back to the original dimensionality by a 'down projection'.

    Attributes:
        gate_proj (nn.Linear):
             The linear transformation layer that projects input from 'dim' to 'hidden_dim' without bias.
        down_proj (nn.Linear):
             The linear transformation layer that projects the gated and activated hidden representation back to 'dim' without bias.
        up_proj (nn.Linear):
             The linear transformation layer that projects input from 'dim' to 'hidden_dim' without bias, used for the gated mechanism.

    Args:
        dim (int):
             The dimensionality of the input features.
        hidden_dim (int):
             The dimensionality of the hidden layer features.

    Returns:
        (mx.array):
             The output of the MLP after the input 'x' is processed through the layers.

    """

    def __init__(self, dim, hidden_dim):
        """
        Initializes a neural network module with specified input and hidden dimensions.

        Args:
            dim (int):
                 The size of the input features.
            hidden_dim (int):
                 The size of the hidden layer features.
            This initializer sets up three linear transformation projects within the neural network:
                 `gate_proj` for transforming input features to hidden layer features without bias, `down_proj` for transforming the hidden layer features back to the input feature size without bias, and `up_proj` for transforming the input features to hidden layer features without bias.

        """
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        """
        Applies a gated projection transformation to the input data.
        This method takes an input x, applies a gating projection via the gate_proj attribute followed by a GELU activation function. The result is then multiplied element-wise with the output of the up_proj attribute applied to the input. The final result is then downsampled using the down_proj attribute.

        Args:
            x (mx.ndarray):
                 The input data to be transformed.

        Returns:
            (mx.ndarray):
                 The transformed output after gated projection and resampling.

        Raises:
            TypeError:
                 If the input data is not of the expected type (mx.ndarray).

        """
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """
    A PyTorch nn.Module that implements a single transformer block used in transformer models.
    This class serves as a building block for transformer architectures and encapsulates a
    single transformer layer. It includes a self-attention mechanism, position-wise feedforward
    network (MLP), and layer normalization steps, encapsulated as class members.

    Attributes:
        num_attention_heads (int):
             The number of attention heads in the self-attention mechanism.
        hidden_size (int):
             The size of the hidden layer.
        self_attn (Attention):
             The self-attention mechanism within the block.
        mlp (MLP):
             The position-wise feedforward network.
        input_layernorm (RMSNorm):
             Layer normalization preceding the self-attention.
        post_attention_layernorm (RMSNorm):
             Layer normalization following the self-attention before the MLP.
        args (TextConfig):
             The configuration object containing various parameters for the transformer block.

    Methods:
        __call__(x, mask=None, cache=None):
            Forward pass through the transformer block.

    Args:
        x (mx.array):
             The input tensor to the transformer block.
        mask (Optional[mx.array]):
             The mask tensor for the self-attention mechanism (default: None).
        cache (Optional[Tuple[mx.array, mx.array]]):
             Tuple containing cached tensors from previous computations which
        can be used in the case of incremental decoding (default:
             None).

    Returns:
        (Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]):
             A tuple consisting of the output tensor of the transformer
            block and an optional cache with tensors for use in subsequent computations.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes the object with the configuration specified in `args`.

        Args:
            args (TextConfig):
                 An instance of `TextConfig` that provides the necessary hyperparameters for initialization. The `TextConfig` should include the number of attention heads, hidden layer size, intermediate layer size, and epsilon value for RMS normalization.

        Raises:
            ValueError:
                 If the provided configuration is invalid, such as mismatched dimensions or missing parameters.

        Attributes:
            num_attention_heads (int):
                 The number of attention heads.
            hidden_size (int):
                 The size of the hidden layers in the network.
            self_attn (Attention):
                 The self-attention module of the network, initialized with the parameters from `args`.
            mlp (MLP):
                 The multi-layer perceptron module, with dimensions specified by `args`.
            input_layernorm (RMSNorm):
                 The layer normalization applied to the input of the attention layer.
            post_attention_layernorm (RMSNorm):
                 The layer normalization applied after the attention calculation and before the output.
            args (TextConfig):
                 A copy of the configuration arguments provided on initialization for future reference.

        """
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """
        Calls the model with input data to perform a forward pass.
        This method accepts an input tensor, an optional mask tensor, and an optional cache.
        It performs a forward pass through a Transformer-like structure using self-attention,
        Layer Normalization, and a feed-forward MLP. It returns the output tensor and updated cache.

        Args:
            x (mx.array):
                 The input tensor to the model.
            mask (Optional[mx.array], optional):
                 The optional mask tensor to be applied during self-attention.
                Defaults to None.
            cache (Optional[Tuple[mx.array, mx.array]], optional):
                 The optional tuple containing tensors to be used as cache
                from previous states. Defaults to None.

        Returns:
            (mx.array):
                 The output tensor after processing input x through the model.
            (Tuple[mx.array, mx.array]):
                 The updated cache after computing the forward pass.

        Raises:
            Custom exceptions are not specified. The behavior is determined by lower-level library implementations.


        """
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class GemmaModel(nn.Module):
    """
    A PyTorch module representing the GemmaModel, which is a transformer model with specific configurations.

    Attributes:
        args (TextConfig):
             Configuration object containing model hyperparameters such as vocab_size,
            num_hidden_layers, hidden_size, and rms_norm_eps.
        vocab_size (int):
             The size of the vocabulary.
        num_hidden_layers (int):
             The number of hidden transformer layers.
        embed_tokens (nn.Embedding):
             Embedding layer for token embedding.
        layers (List[TransformerBlock]):
             A list of transformer blocks constituting the model's layers.
        norm (RMSNorm):
             The root mean square layer normalization applied to the output of the transformer layers.

    Methods:
        __call__(inputs, cache=None, inputs_embeds=None, mask=None):
            Defines the forward pass of the GemmaModel.

    Args:
        inputs (mx.array):
             Input tensor containing token indices which are to be embedded.
        cache (optional):
             Cache containing past hidden states (default is None).
        inputs_embeds (optional):
             Pre-computed embeddings for the inputs. If None, embeddings are computed using inputs (default is None).
        mask (Optional[mx.array], optional):
             An optional mask to hide future tokens during training for autoregressive models (default is None).

    Returns:
        (Tuple[mx.array, List]):
             A tuple containing the normalized output of the last transformer layer and the updated cache.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes a new instance of the object with the provided configuration parameters.

        Args:
            args (TextConfig):
                 An instance of TextArgs, which provides configuration parameters such as vocabulary size, number of hidden layers, hidden state size, and normalization epsilon value.

        Raises:
            AssertionError:
                 If the 'vocab_size' attribute of the args is not greater than 0.

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
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        # for passing merged input embeddings
        """
        Calls the NN model on a set of inputs to obtain outputs.
        This method applies the model's computations to the provided inputs. If the input embeddings are not given, it uses the
        `embed_tokens` method to obtain them from the raw inputs. It scales the embeddings by the square root of the hidden
        size. A causal mask is created and applied if a cache is provided, which is used for autoregressive decoding.
        method iterates over the model's layers, passing through each layer's computations and updating the cache. Finally,
        the output is normalized before being returned.

        Args:
            inputs (mx.array):
                 Raw input tokens to be processed by the model.
            cache (optional):
                 A list of cached states from previous forward passes.
            inputs_embeds (optional):
                 Precomputed embeddings of the inputs. If provided, the raw inputs are ignored.
            mask (Optional[mx.array], optional):
                 Optional mask to be applied to the inputs.

        Returns:
            (tuple):
                 A tuple containing the normalized output of the final layer and the updated cache.

        """
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)

        else:
            h = inputs_embeds

        h = h * (self.args.hidden_size**0.5)

        if cache is not None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache


class LanguageModel(nn.Module):
    """
    A class representing a language model based on the GemmaModel architecture.
    This class is designed to encapsulate a language model with functionalities to process inputs, generate outputs,
    and sanitize model weights. It inherits from nn.Module, allowing it to integrate seamlessly with PyTorch's
    modeling framework.

    Attributes:
        args (TextConfig):
             Configuration object containing model arguments.
        model_type (str):
             The type of model specified in the configuration.
        model (GemmaModel):
             The underlying GemmaModel instance.

    Methods:
        __init__(self, args:
             TextConfig):
            Initializes a new instance of LanguageModel.
        __call__(self, inputs:
             mx.array, cache=None, inputs_embeds=None, mask: Optional[mx.array]=None):
            Allows the LanguageModel to be called as a function, processing the provided inputs and optional arguments,
            returning the model's output and updated cache.
        sanitize(self, weights:
             dict) -> dict:
            Cleans the provided weights dictionary by removing entries associated with the rotary position embeddings.
            @property
        layers(self):
            Property that returns the layers of the underlying GemmaModel.

    """

    def __init__(self, args: TextConfig):
        """
        Initializes a new instance of a neural network model wrapper class.

        Args:
            args (TextConfig):
                 An instance of TextConfig that contains the configuration settings used
                to initialize the model. This includes the model type and various hyperparameters necessary
                for model construction.

        Attributes:
            args (TextConfig):
                 Stores the configuration settings supplied during instantiation.
            model_type (str):
                 Stores the type of the model which can be determined from the args.
            model (GemmaModel):
                 The actual neural network model created using the configuration
                provided in args. It is an instance of the GemmaModel class.

        """
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GemmaModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        """
        Generates output from the model given various input and contextual parameters.
        This method wraps the model's functionality, allowing you to pass in data and optionally
        a cache and input embeddings. It applies the model to the inputs to generate output,
        which could be used for tasks such as language modeling or text generation. The output
        is further processed by converting it to a linear space using the embedding tokens from
        the model.

        Args:
            inputs (mx.array):
                 The input data to the model, typically token indices.
            cache (optional):
                 The cache containing intermediate computational states to allow
                for efficient decoding. Defaults to None.
            inputs_embeds (optional):
                 Precomputed embeddings for the input data. If provided,
                it bypasses the model's own embedding generation.
                Defaults to None.
            mask (Optional[mx.array]):
                 An optional mask to be applied to the inputs,
                which can be used to ignore certain parts of the input
                for attention purposes. Defaults to None.

        Returns:
            (tuple):
                 A tuple containing the output of the model and the updated cache.
                The output is also transformed into a linear space using the token
                embeddings from the model.

        """
        out, cache = self.model(inputs, cache, inputs_embeds=inputs_embeds, mask=mask)
        out = self.model.embed_tokens.as_linear(out)
        return out, cache

    def sanitize(self, weights):
        """
        Sanitizes the given weights dictionary by removing the key-value pair where the key contains specific substrings.
        This function iterates over the dictionary of weights and filters out entries with keys that include the substring
        'self_attn.rotary_emb.inv_freq'. The purpose of the filter is to exclude specific configuration values or weights
        from a model's parameters, potentially prior to model loading or saving. The resulting dictionary is free of the specified
        entries, and can be considered sanitized.

        Args:
            self:
                 The class instance in which this method is defined. Typically, this might be an instance of a model-related class.
            weights (dict):
                 A dictionary where keys are strings representing the parameter names, and values are the parameters themselves.

        Returns:
            (dict):
                 A new dictionary containing only the key-value pairs where the key does not include the specified substring.

        """
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        """
        Gets the layers of the model.
        This property returns the list of all layers within the model.

        Returns:
            (List):
                 A list containing the model's layers.

        """
        return self.model.layers
