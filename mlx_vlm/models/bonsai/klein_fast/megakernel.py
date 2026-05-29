from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

from mlx_vlm.models.bonsai.klein_fast.blocks import (
    DenseLinearKernel,
    DoubleBlockWeights,
    DoubleFlux2Block,
    Flux2KleinBlockSpec,
    QuantizedLinearKernel,
    SingleBlockWeights,
    SingleFlux2Block,
    _layer_norm_affine,
    _silu,
)

PrecisionName = Literal["bf16", "1bit", "2bit"]


@dataclass(frozen=True)
class Flux2KleinMegakernelSpec:
    num_double_blocks: int = 5
    num_single_blocks: int = 20
    dim: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    mlp_ratio: float = 3.0
    layer_norm_eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    rope_theta: int = 2000
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    in_channels: int = 128
    context_dim: int = 7680

    @property
    def block_spec(self) -> Flux2KleinBlockSpec:
        return Flux2KleinBlockSpec(
            dim=self.dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            axes_dims_rope=self.axes_dims_rope,
        )


@dataclass
class MegakernelWeights:
    x_embedder: mx.array
    context_embedder: mx.array
    norm_out_linear: mx.array
    proj_out: mx.array
    double_block_weights: list[DoubleBlockWeights]
    single_block_weights: list[SingleBlockWeights]


class Flux2KleinMegakernel:
    def __init__(
        self,
        *,
        spec: Flux2KleinMegakernelSpec,
        weights: MegakernelWeights,
        precision: PrecisionName,
        group_size: int,
    ):
        if len(weights.double_block_weights) != spec.num_double_blocks:
            raise ValueError(
                f"Expected {spec.num_double_blocks} double block weight sets, got {len(weights.double_block_weights)}"
            )
        if len(weights.single_block_weights) != spec.num_single_blocks:
            raise ValueError(
                f"Expected {spec.num_single_blocks} single block weight sets, got {len(weights.single_block_weights)}"
            )

        self.spec = spec
        block_spec = spec.block_spec

        self.x_embedder = DenseLinearKernel(weights.x_embedder)
        self.context_embedder = DenseLinearKernel(weights.context_embedder)
        self.norm_out_linear = DenseLinearKernel(weights.norm_out_linear)
        self.proj_out = DenseLinearKernel(weights.proj_out)

        self.double_blocks = [
            DoubleFlux2Block(
                spec=block_spec,
                weights=w,
                precision=precision,
                group_size=group_size,
            )
            for w in weights.double_block_weights
        ]
        self.single_blocks = [
            SingleFlux2Block(
                spec=block_spec,
                weights=w,
                precision=precision,
                group_size=group_size,
            )
            for w in weights.single_block_weights
        ]

    def prepare_all_modulations(self, temb: mx.array) -> dict:
        """Pre-compute all 25 block modulations + norm_out affine params.

        Returns a dict of flattened modulation tuples suitable for use as
        closure constants in a compiled forward pass.
        """
        if temb.ndim == 1:
            temb = temb[None, :]

        double_mods = []
        for block in self.double_blocks:
            img_mod = block.prepare_modulation(block.modulation_img, temb)
            txt_mod = block.prepare_modulation(block.modulation_txt, temb)
            double_mods.append((img_mod, txt_mod))

        single_mods = [block.prepare_modulation(temb) for block in self.single_blocks]

        norm_mod = self.norm_out_linear(_silu(temb))
        norm_scale, norm_shift = mx.split(norm_mod, 2, axis=-1)
        norm_w = (1.0 + norm_scale).reshape(-1)
        norm_b = norm_shift.reshape(-1)

        return {
            "double_mods": double_mods,
            "single_mods": single_mods,
            "norm_w": norm_w,
            "norm_b": norm_b,
        }

    def forward(
        self,
        latents: mx.array,
        text_embeddings: mx.array,
        temb: mx.array,
        rotary_cos: mx.array,
        rotary_sin: mx.array,
    ) -> mx.array:
        img = self.x_embedder(latents)
        txt = self.context_embedder(text_embeddings)
        text_seq_len = int(txt.shape[1])

        for block in self.double_blocks:
            txt, img = block.forward(img, txt, temb, rotary_cos, rotary_sin)

        hidden = mx.concatenate([txt, img], axis=1)

        for block in self.single_blocks:
            hidden = block.forward(hidden, temb, rotary_cos, rotary_sin)

        img_out = hidden[:, text_seq_len:, :]

        mod = self.norm_out_linear(_silu(temb))
        scale, shift = mx.split(mod, 2, axis=-1)
        img_out = mx.fast.layer_norm(
            img_out,
            (1.0 + scale).reshape(-1),
            shift.reshape(-1),
            self.spec.layer_norm_eps,
        )

        return self.proj_out(img_out)

    def forward_from_modulations(
        self,
        latents: mx.array,
        text_embeddings: mx.array,
        precomputed: dict,
        rotary_cos: mx.array,
        rotary_sin: mx.array,
    ) -> mx.array:
        double_mods = precomputed["double_mods"]
        single_mods = precomputed["single_mods"]
        norm_w = precomputed["norm_w"]
        norm_b = precomputed["norm_b"]

        img = self.x_embedder(latents)
        txt = self.context_embedder(text_embeddings)
        text_seq_len = int(txt.shape[1])

        for i, block in enumerate(self.double_blocks):
            img_mod, txt_mod = double_mods[i]
            txt, img = block.forward_from_modulation(
                img, txt, img_mod, txt_mod, rotary_cos, rotary_sin
            )

        hidden = mx.concatenate([txt, img], axis=1)

        for i, block in enumerate(self.single_blocks):
            hidden = block.forward_from_modulation(
                hidden, single_mods[i], rotary_cos, rotary_sin
            )

        img_out = hidden[:, text_seq_len:, :]
        img_out = _layer_norm_affine(img_out, norm_w, norm_b, self.spec.layer_norm_eps)
        return self.proj_out(img_out)


def eval_megakernel_constants(model: Flux2KleinMegakernel) -> None:
    """Materialize all lazy weight arrays so they are not rebuilt per call.

    Useful before benchmarking or serving: ensures quantized packed weights,
    scales, biases, and RMSNorm gains are realized on-device.
    """
    arrays: list[mx.array] = []

    def _collect_linear(linear):
        if isinstance(linear, QuantizedLinearKernel):
            arrays.extend([linear.packed_weight, linear.scales, linear.biases])
        elif isinstance(linear, DenseLinearKernel):
            arrays.append(linear.weight)

    _collect_linear(model.x_embedder)
    _collect_linear(model.context_embedder)
    _collect_linear(model.norm_out_linear)
    _collect_linear(model.proj_out)

    for block in [*model.double_blocks, *model.single_blocks]:
        for attr_name in dir(block):
            if attr_name.startswith("_"):
                continue
            attr = getattr(block, attr_name)
            if isinstance(attr, mx.array):
                arrays.append(attr)
            elif isinstance(attr, (QuantizedLinearKernel, DenseLinearKernel)):
                _collect_linear(attr)

    if arrays:
        mx.eval(*arrays)
