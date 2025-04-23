import math
from typing import Union

import mlx.core as mx
import mlx.nn as nn


class LoraAdapter(nn.Module):
    def __init__(self, input_dims, rank, std_dev=None):
        super().__init__()
        if std_dev is None:
            self.weight = mx.ones((input_dims, rank))
        else:
            self.weight = mx.random.uniform(
                low=-std_dev,
                high=std_dev,
                shape=(input_dims, rank),
            )


class LoRaLayer(nn.Module):
    def __init__(
        self,
        linear: Union[nn.Linear, nn.QuantizedLinear],
        rank: int,
        alpha: float = 0.1,
        dropout: float = 0.0,
        name: str = None,
        disable_adapter: bool = False,
    ):
        super().__init__()
        self.disable_adapter = disable_adapter

        self.base_layer = linear
        self.dropout = nn.Dropout(p=dropout)

        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        std_dev = 1 / math.sqrt(rank)

        # Create A and B as proper nested modules
        if name is not None:
            self.lora_name = name
            self.lora_A = nn.Module()
            self.lora_B = nn.Module()
            setattr(self.lora_A, name, LoraAdapter(input_dims, rank))
            setattr(self.lora_B, name, LoraAdapter(rank, output_dims))

        else:
            self.A = mx.random.uniform(
                low=-std_dev,
                high=std_dev,
                shape=(input_dims, rank),
            )
            self.B = mx.ones((rank, output_dims))

        self.alpha = alpha

    def __call__(self, x):
        y = self.base_layer(x)

        if self.disable_adapter:
            return y

        if hasattr(self, "lora_name"):
            A = getattr(self.lora_A, self.lora_name).weight
            B = getattr(self.lora_B, self.lora_name).weight

            # Dimensions for x @ A to work:
            if x.shape[-1] == A.shape[1]:
                A_transposed = A.T
                B_transposed = B.T
            else:
                A_transposed = A
                B_transposed = B

            lora_update = (self.dropout(x) @ A_transposed) @ B_transposed
        else:
            lora_update = (self.dropout(x) @ self.A) @ self.B

        return y + (self.alpha * lora_update).astype(x.dtype)


def replace_lora_with_linear(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, LoRaLayer):
            # Compute the final merged weight
            lora_update = layer.alpha * (layer.A @ layer.B)
            updated_weight = layer.original_layer.weight + lora_update
            use_bias = layer.original_layer.bias is not None

            updated_bias = layer.original_layer.bias

            # Create a new Linear layer with the updated parameters
            new_linear_layer = nn.Linear(
                updated_weight.size(1), updated_weight.size(0), bias=use_bias
            )

            new_linear_layer.weight = updated_weight

            if use_bias:
                new_linear_layer.bias = updated_bias

            if isinstance(layer.original_layer, nn.QuantizedLinear):
                new_linear_layer = nn.QuantizedLinear.from_linear(
                    new_linear_layer,
                    new_linear_layer.group_size,
                    new_linear_layer.bits,
                )

            # Replace the LoRaLayer with the new Linear layer in the model
            model.layers[i] = new_linear_layer
