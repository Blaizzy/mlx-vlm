from typing import Any, Dict, Literal, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Special token id for audio
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'


class DepthWiseSeperableConv1d(nn.Module):

    def __init__(
        self,
        input_dim,
        depthwise_seperable_out_channel,
        kernel_size,
        depthwise_multiplier,
        padding=0,
    ):
        super().__init__()

        self.dw_conv = nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            1,
            padding=padding,
            groups=input_dim,
        )

        if depthwise_seperable_out_channel != 0:
            self.pw_conv = nn.Conv1d(
                input_dim * depthwise_multiplier,
                depthwise_seperable_out_channel,
                1,
                1,
                0,
            )
        else:
            self.pw_conv = nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def __call__(self, x):
        """
        Args:
            x: torch.Tensor
                input tensor
        """
        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))

        # Running statistics
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x):
        # x shape: (batch_size, num_features, seq_len)
        # Compute statistics along batch and sequence dimensions
        mean = mx.mean(x, axis=(0, 2), keepdims=True)
        var = mx.var(x, axis=(0, 2), keepdims=True)

        # Update running statistics
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean.squeeze()
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * var.squeeze()

        # Normalize
        x_norm = (x - mean) / mx.sqrt(var + self.eps)

        # Apply learnable parameters
        return self.weight.reshape(1, -1, 1) * x_norm + self.bias.reshape(1, -1, 1)


# TODO: Improve using GLU class
class GLUPointWiseConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        glu_type="sigmoid",
        bias_in_glu=True,
        causal=False,
    ):
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        if causal:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim, output_dim * 2, kernel_size, 1, padding=(kernel_size - 1)
            )
        else:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_act}")

        if bias_in_glu:
            self.b1 = mx.zeros((1, output_dim, 1))
            self.b2 = mx.zeros((1, output_dim, 1))

    def __call__(self, x):
        # to be consistent with GLULinear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = x.transpose((0, 2, 1))
        x = self.ext_pw_conv_1d(x)
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * (
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * (
                    x[:, self.output_dim : self.output_dim * 2, :]
                )
        else:
            if self.bias_in_glu:
                x = (x[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :] + self.b2
                )
            else:
                x = (x[:, 0 : self.output_dim, :]) * self.glu_act(
                    x[:, self.output_dim : self.output_dim * 2, :]
                )

        x = x.transpose((0, 2, 1))
        return x


class ConformerConvModule(nn.Module):

    def __init__(
        self,
        input_dim,
        ext_pw_out_channel,
        depthwise_seperable_out_channel,
        ext_pw_kernel_size,
        kernel_size,
        depthwise_multiplier,
        dropout_rate,
        causal=False,
        batch_norm=False,
        chunk_se=0,
        chunk_size=18,
        activation="relu",
        glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        export=False,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal

        self._add_ext_pw_layer()

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        if batch_norm:

            self.bn_layer = BatchNorm1d(input_dim)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Activation function {activation} not supported")

        self.dropout = nn.Dropout(dropout_rate)
        self.export = export

        if causal:
            if export:  # Inference only.
                padding = (
                    0  # A cache is concatenated to the left. No padding in the kernel.
                )
            else:
                # Training only. Padding will be added symmetrically on both sides.
                # After convolution, clip off kernel_size-1 points on the right.
                padding = kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = nn.Linear(depthwise_seperable_out_channel, input_dim)
        else:
            if depthwise_multiplier != 1:
                self.ln2 = nn.Linear(input_dim * depthwise_multiplier, input_dim)

    def _add_ext_pw_layer(self):
        """
        This function is an extension of __init__ function
        and dedicated to the convolution module creation
        of the conformer.
        """
        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = (
            nn.Identity()
        )  # jit hacks.
        self.squeeze_excitation = nn.Identity()  # jit.
        self.apply_ln1 = self.fix_len1 = False  # jit.

        if self.ext_pw_out_channel != 0:
            if self.causal:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1),
                )
                if self.ext_pw_kernel_size > 1:
                    self.fix_len1 = True
                else:
                    self.fix_len1 = False
            else:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1) // 2,
                )
                self.fix_len1 = False

            if self.linear_glu_in_convm:
                self.glu = GLULinear(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.glu_type,
                    self.bias_in_glu,
                )
            else:
                self.glu = GLUPointWiseConv(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    self.glu_type,
                    self.bias_in_glu,
                    self.causal,
                )

            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = mx.ones((3,))
            self.pw_conv_simplify_b = mx.zeros((3,))

    def __call__(self, x):
        """ConvModule Forward.
        Args:
            x: torch.Tensor
                input tensor.
        """
        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, : -(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1

        x = x.transpose((0, 2, 1))

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.transpose((0, 2, 1))
            x = self.ln2(x)
            x = x.transpose((0, 2, 1))
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, : -(self.ext_pw_kernel_size - 1)]

            if self.apply_ln1:
                x = x.transpose((0, 2, 1))
                x = self.ln1(x)
                x = x.transpose((0, 2, 1))

            x = x.transpose((0, 2, 1))
        else:
            x = x.unsqueeze(1).transpose((0, 1, 3, 2))
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = x.squeeze(1)

        x = self.dropout(x)
        return x


class GLU(nn.Module):
    """Implement Gated Linear Unit (GLU) module"""

    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()

        if self.act_name == "relu":
            self.act_fn = nn.ReLU()
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = nn.SiLU()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        half_x, gate = mx.split(x, 2, axis=self.dim)
        return half_x * self.act_fn(gate)


class GLULinear(nn.Module):
    """Linear + GLU module
    Args:
        input_dim: int
            input size
        output_dim: int
            output size.
        glu_type:
            activation function name used in glu module.
            default "sigmoid" (swish function).
        bias_in_glu: bool, optional
            If True, the addtive bias is added. Default False.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        glu_type="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type)

    def __call__(self, x):
        """GLULinear forward
        Args:
            x: torch.Tensor
                inpute tensor.
        """
        x = self.linear(x)
        return self.glu_act(x)


class FeedForward(nn.Module):
    """Feed Forward module for Conformer."""

    def __init__(
        self,
        d_model,
        d_inner,
        dropout_rate,
        activation="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, bias_in_glu=True)
        self.net = [
            module,
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate),
        ]

    def __call__(self, x):
        # Layer normalization
        x = self.layer_norm(x)
        for layer in self.net:
            x = layer(x)
        return x


class ConformerAttention(nn.Module):
    """Multi-headed attention module for Conformer."""

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        attention_inner_dim=-1,
        glu_type="swish",
        bias_in_glu=True,
        use_pt_scaled_dot_product_attention=False,
        n_value=-1,
        group_size: int = 1,
    ):
        super().__init__()

        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = attention_inner_dim // n_head
        self.scale = self.d_k**-0.5
        self.h = n_head
        assert n_head % group_size == 0, "group_size must divide n_head"
        self.g = group_size
        self.h_k = n_head // group_size

        self.linear_q = nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_out = nn.Linear(attention_inner_dim // group_size, n_feat)
        self.dropout = dropout_rate

    def __call__(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        q = (
            self.linear_q(query)
            .reshape((batch_size, seq_len, self.heads, -1))
            .transpose((0, 2, 1, 3))
        )
        k = (
            self.linear_k(key)
            .reshape((batch_size, seq_len, self.heads, -1))
            .transpose((0, 2, 1, 3))
        )
        v = (
            self.linear_v(value)
            .reshape((batch_size, seq_len, self.heads, -1))
            .transpose((0, 2, 1, 3))
        )

        # Compute attention scores
        attention_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting with attention scores
            mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            attention_scores = mx.where(mask, attention_scores, mx.array(float("-inf")))

        # Apply softmax to get attention weights
        attention_weights = nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights during training
        if self.dropout > 0:
            attention_weights = nn.Dropout(self.dropout)(attention_weights)

        # Apply attention weights to values
        context = mx.matamul(attention_weights, v)

        # Transpose and reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Final projection
        output = self.linear_out(context)

        return output


class ConformerEncoderLayer(nn.Module):
    """A single Conformer block."""

    def __init__(
        self,
        d_model=512,
        ext_pw_out_channel=0,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        n_head=4,
        d_ffn=2048,
        ext_pw_kernel_size=1,
        kernel_size=3,
        dropout_rate=0.1,
        causal=False,
        batch_norm=False,
        activation="relu",
        chunk_se=0,
        chunk_size=18,
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_innner_dim=-1,
        attention_glu_type="swish",
        activation_checkpointing="",
        export=False,
        use_pt_scaled_dot_product_attention=False,
        attn_group_sizes: int = 1,
    ):
        super().__init__()

        self.feed_forward_in = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.self_attn = ConformerAttention(
            n_head,
            d_model,
            dropout_rate,
            attention_innner_dim,
            attention_glu_type,
            bias_in_glu,
            use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
            group_size=attn_group_sizes,
        )
        self.conv = ConformerConvModule(
            d_model,
            ext_pw_out_channel,
            depthwise_seperable_out_channel,
            ext_pw_kernel_size,
            kernel_size,
            depthwise_multiplier,
            dropout_rate,
            causal,
            batch_norm,
            chunk_se,
            chunk_size,
            conv_activation,
            conv_glu_type,
            bias_in_glu,
            linear_glu_in_convm,
            export=export,
        )
        self.feed_forward_out = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.layer_norm_att = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def __call__(self, x, mask=None):
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)

        x = x + self.self_attn(norm_x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)
        return out


class ConformerEncoder(nn.Module):
    """Conformer encoder for audio processing."""

    def __init__(
        self,
        input_size,
        chunk_size,
        left_chunk,
        num_lang=None,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        input_layer="nemo_conv",
        causal=True,
        batch_norm=False,
        cnn_out=-1,
        cnn_layer_norm=False,
        ext_pw_out_channel=0,
        ext_pw_kernel_size=1,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        chunk_se=0,
        kernel_size=3,
        activation="relu",
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_glu_type="swish",
        export=False,
        extra_layer_output_idx=-1,
        extra_multi_layer_output_idxs=[],
        activation_checkpointing="",
        relative_attention_bias_args=None,
        time_reduction=4,
        use_pt_scaled_dot_product_attention=False,
        nemo_conv_settings=None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none", True] = "none",
        replication_pad_for_subsample_embedding=False,
        attention_group_size=1,
        encoder_embedding_config=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.attention_dim = attention_dim

        # Input layer
        if input_layer == "conv2d":
            self.embed = [
                nn.Conv2d(1, attention_dim // 4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(
                    attention_dim // 4,
                    attention_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    attention_dim // 2,
                    attention_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
            ]
        else:
            self.embed = [nn.Linear(input_size, attention_dim)]

        # Positional encoding - using sinusoidal positional embedding
        # In a complete implementation, we would use a proper positional encoding here

        # Conformer blocks
        self.encoders = [
            ConformerEncoderLayer(
                d_model=attention_dim,
                ext_pw_out_channel=ext_pw_out_channel,
                depthwise_seperable_out_channel=depthwise_seperable_out_channel,
                depthwise_multiplier=depthwise_multiplier,
                n_head=attention_heads,
                d_ffn=linear_units,
                ext_pw_kernel_size=ext_pw_kernel_size,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                causal=causal,
                batch_norm=batch_norm,
                activation=activation,
                chunk_se=chunk_se,
                chunk_size=chunk_size,
                conv_activation=conv_activation,
                conv_glu_type=conv_glu_type,
                bias_in_glu=bias_in_glu,
                linear_glu_in_convm=linear_glu_in_convm,
                attention_glu_type=attention_glu_type,
                # activation_checkpointing=attn_checkpointing(activation_checkpointing, i),
                export=export,
                use_pt_scaled_dot_product_attention=use_pt_scaled_dot_product_attention,
                attn_group_sizes=attention_group_size,
            )
            for _ in range(num_blocks)
        ]

        self.norm = nn.LayerNorm(attention_dim)

    def post_init(self, init_model=None):
        """Initialize model weights from a pretrained model."""
        # In a complete implementation, this would load weights from a checkpoint
        pass

    def __call__(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, time_length, feat_dim)
            mask: Mask tensor for valid data (batch_size, time_length)

        Returns:
            x: Encoded features (batch_size, new_length, attention_dim)
            mask: Mask for the encoded features
        """

        # Handle input layer based on its type
        if len(self.embed) > 1:  # conv2d case
            # Add channel dimension: (batch_size, time_length, feat_dim) -> (batch_size, 1, time_length, feat_dim)
            x = mx.expand_dims(x, axis=1)

            # Apply convolutional layers
            for layer in self.embed[:2]:
                x = layer(x)

            # For each conv layer, the sequence length is reduced
            if mask is not None:
                # Downsample the mask to match the reduced sequence length
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            for layer in self.embed[2:4]:
                x = layer(x)

            # Downsample mask again
            if mask is not None:
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            for layer in self.embed[4:]:
                x = layer(x)

            # Downsample mask once more
            if mask is not None:
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            # Rearrange from (batch_size, channels, time, freq) -> (batch_size, time, channels*freq)
            batch_size, channels, time, freq = x.shape
            x = mx.transpose(x, (0, 2, 1, 3))
            x = mx.reshape(x, (batch_size, time, channels * freq))

        else:  # Linear case
            x = self.embed[0](x)

        # Apply Conformer blocks
        for encoder in self.encoders:
            x = encoder(x, mask)

        # Final normalization
        x = self.norm(x)

        return x, mask


class AudioModel(nn.Module):
    """Audio embedding."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # Get hidden size for text LM
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size

        # Dropout setting
        self.drop = None
        if hasattr(config, "embd_pdrop") or hasattr(config, "embed_pdrop"):
            embd_drop = (
                config.embd_pdrop
                if hasattr(config, "embd_pdrop")
                else config.embed_pdrop
            )
            self.embd_drop = embd_drop  # Store dropout rate

        # Configure audio processor
        if (
            isinstance(config.audio_processor, dict)
            and config.audio_processor.get("name", None) == "cascades"
        ):
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None

            # Create encoder
            self.encoder = ConformerEncoder(**encoder_config)

            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError("Audio processor not implemented")

        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        # Configuration
        self.freeze_audio_processor = kwargs.get("freeze_audio_processor", False)
        self.downsample_rate = kwargs.get("downsample_rate", 1)

        # Projection layer
        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == "mlp":
            # Follow llava-v1.5's implementation
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = self.downsample_rate

            # Create projection for speech mode
            layers_for_speech = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers_for_speech.extend(
                    [nn.GELU(), nn.Linear(dim_projection, dim_projection)]
                )

            audio_projection_for_speech = layers_for_speech

            # Create projection for vision mode
            layers_for_vision = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers_for_vision.extend(
                    [nn.GELU(), nn.Linear(dim_projection, dim_projection)]
                )

            audio_projection_for_vision = layers_for_vision

            # Store as a dictionary
            self.audio_projection = {
                "speech": audio_projection_for_speech,
                "vision": audio_projection_for_vision,
            }
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}")

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        """Initialize the audio encoder with pretrained weights."""
        if audio_config.get("name", None) == "cascades":
            init_model_config = audio_config.get("init_model", {})
            self.encoder.post_init(init_model_config)

            # Remove init_model to save memory
            if "init_model" in audio_config:
                audio_config.pop("init_model")

    def set_audio_embeds(self, input_embeds):
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes):
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(
        self, input_embeds, audio_attention_mask, audio_projection_mode="speech"
    ):
        """Process audio inputs through the encoder and projection layers."""

        # Apply encoder with or without gradient based on freeze setting
        if self.freeze_audio_processor:
            # In MLX, we would implement a mechanism to stop gradient flow
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        # Apply projection based on its type
        if isinstance(self.audio_projection, dict):
            # Sequential projection for the specified mode
            projection_layers = self.audio_projection[audio_projection_mode]

            # Apply the layers in sequence
            audio_set_tensor = audio_features
            for layer in projection_layers:
                audio_set_tensor = layer(audio_set_tensor)
        else:
            # Single linear projection
            audio_set_tensor = self.audio_projection(audio_features)

        return audio_set_tensor

    def __call__(
        self,
        input_ids,
        input_embeds,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        **kwargs,
    ):
        """
        Forward pass for audio embeddings.

        Args:
            input_ids: Input text ids (B, U)
            input_embeds: Audio features (B, T, D)  B: num audios in a sequence
        """
        # Use cached inputs if available
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.copy()
            self.input_embeds = None

        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.copy()
            self.audio_embed_sizes = None

        # Reshape input_ids if needed
        input_shape = input_ids.shape
        input_ids = mx.reshape(input_ids, (-1, input_shape[-1]))

        # Find positions of audio token IDs
        positions = mx.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID)

        # Determine target device and dtype from projection layer
        if isinstance(self.audio_projection, dict):
            target_dtype = mx.float32  # Default dtype
        else:
            target_dtype = mx.float32

        # Convert input_embeds to target dtype if available
        if input_embeds is not None:
            input_embeds = input_embeds.astype(target_dtype)

        # Process audio if audio tokens are present
        if len(positions) > 0:
            audio_set_tensor = self.get_audio_features(
                input_embeds, audio_attention_mask, audio_projection_mode
            )
        else:
            # Create dummy audio tensor for training
            if True:  # Equivalent to self.training in PyTorch
                audio_embeds = mx.zeros((1, 500, self.audio_dim_in), dtype=target_dtype)
                audio_attention_mask = mx.ones((1, 500), dtype=mx.int32)
                audio_set_tensor = self.get_audio_features(
                    audio_embeds, audio_attention_mask, audio_projection_mode
                )

        # Get token embeddings
        hidden_states = kwargs["wte"](input_ids)

        if len(positions) > 0:
            # Validate that we have correct number of positions
            assert audio_embed_sizes.sum().item() == len(
                positions
            ), f"Number of encoder outputs ({audio_embed_sizes.sum().item()}) must match number of audio tokens ({len(positions)})"

            # Create a list of audio features based on sizes
            merged_audio_set_tensor = []
            start_idx = 0
            for i in range(len(audio_embed_sizes)):
                size = audio_embed_sizes[i]
                merged_audio_set_tensor.append(audio_set_tensor[i, :size])
                start_idx += size

            # Concatenate all features
            merged_audio_set_tensor = mx.concatenate(merged_audio_set_tensor, axis=0)
            merged_audio_set_tensor = merged_audio_set_tensor.astype(
                hidden_states.dtype
            )

            # Create a new hidden_states with audio embeddings inserted
            for i, pos in enumerate(positions):
                batch_idx, seq_idx = pos
                # Update the tensor at the specified position
                hidden_states = mx.indexed_update(
                    hidden_states,
                    ((batch_idx, seq_idx),),
                    mx.expand_dims(merged_audio_set_tensor[i], axis=0),
                )
        else:
            # For training with no audio tokens, add a small contribution to maintain gradient flow
            if True:  # Equivalent to self.training
                hidden_states = hidden_states + (0 * audio_set_tensor[:, 0]).sum()

        # Apply dropout if configured
        if self.drop is not None:
            hidden_states = nn.Dropout(self.embd_drop)(hidden_states)

        return hidden_states

    def sanitize(self, weights):
        return weights
