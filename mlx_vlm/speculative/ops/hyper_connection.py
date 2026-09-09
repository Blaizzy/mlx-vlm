"""Decode-equivalent hyperconnection operations shared by target adapters."""

import mlx.core as mx

from ...models.deepseek_v4.hyper_connection import (
    HyperConnection,
    _hc_kernel,
    hc_expand,
)
from .glm5_next import exact_hc_expand, exact_hc_norm, exact_hc_normalized_norm


class HyperConnectionOps:
    @staticmethod
    def _hc_expand(x, residual, post, comb):
        output = exact_hc_expand(x, residual, post, comb)
        if output is not None:
            return output
        return hc_expand(x, residual, post, comb)

    @staticmethod
    def _hc_inputs(connection, x):
        y = x.astype(mx.float32)
        if x.shape[1] <= 1 or x.shape[0] > 1:
            normalized = mx.fast.rms_norm(
                y.flatten(-2),
                None,
                connection.norm_eps,
            )
            return y, normalized @ connection.fn.T

        # At B=1, flattening verifier time changes MLX's FP32 matmul reduction
        # order. Keep each mix projection decode-shaped while sharing the
        # Sinkhorn/collapse work across the complete verifier block.
        mixes = []
        for index in range(x.shape[1]):
            normalized = mx.fast.rms_norm(
                y[:, index : index + 1].flatten(-2),
                None,
                connection.norm_eps,
            )
            mixes.append(normalized @ connection.fn.T)
        return y, mx.concatenate(mixes, axis=1)

    def _hc(self, connection, x):
        if _hc_kernel is None or connection.hc_mult != 4:
            return connection(x)
        y, mixes = self._hc_inputs(connection, x)
        return _hc_kernel(
            x,
            y,
            mixes,
            connection.scale,
            connection.base,
            connection.hc_mult,
            connection.sinkhorn_iters,
            connection.hc_eps,
        )

    def _hc_norm(self, connection, norm, x):
        if x.shape[0] == 1:
            output = exact_hc_normalized_norm(connection, norm, x)
            if output is not None:
                return output
        if _hc_kernel is not None:
            _y, mixes = self._hc_inputs(connection, x)
            output = exact_hc_norm(connection, norm, x, mixes)
            if output is not None:
                return output
        collapsed, post, comb = self._hc(connection, x)
        return norm(collapsed), post, comb


_OPS = HyperConnectionOps()


class SpeculativeHyperConnection(HyperConnection):
    def apply_branch(self, x, norm, branch, *args, **kwargs):
        collapsed, post, comb = _OPS._hc_norm(self, norm, x)
        output = branch(collapsed, *args, **kwargs)
        return _OPS._hc_expand(output, x, post, comb)
