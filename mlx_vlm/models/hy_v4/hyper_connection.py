import mlx.core as mx
import mlx.nn as nn


class IdentityHyperConnectionPre(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_magnitude = config.hc_magnitude
        self.norm_eps = config.rms_norm_eps
        self.hc_fn = mx.zeros(
            (2 * self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.hc_base = mx.zeros((2 * self.hc_mult,), dtype=mx.float32)
        self.hc_scale = mx.ones((2,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        residual = x.astype(mx.float32)
        normalized = mx.fast.rms_norm(residual.flatten(-2), None, self.norm_eps)
        mixes = normalized @ self.hc_fn.T
        pre, post = mx.split(mixes, 2, axis=-1)
        pre_base, post_base = mx.split(self.hc_base, 2, axis=-1)
        pre = mx.sigmoid(pre * self.hc_scale[0] + pre_base) + self.hc_eps
        post = self.hc_magnitude * mx.sigmoid(post * self.hc_scale[1] + post_base)
        collapsed = (pre[..., None] * residual).sum(axis=2).astype(x.dtype)
        return collapsed, post


class IdentityHyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_pre = IdentityHyperConnectionPre(config)

    def __call__(self, x: mx.array):
        return self.hc_pre(x)


def hc_expand(x: mx.array, residual: mx.array, post: mx.array) -> mx.array:
    return (
        residual.astype(mx.float32)
        + post[..., None] * x[:, :, None, :].astype(mx.float32)
    ).astype(residual.dtype)


class IdentityHyperHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hc_head_fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.hc_head_base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.hc_head_scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x.astype(mx.float32)
        normalized = mx.fast.rms_norm(residual.flatten(-2), None, self.norm_eps)
        mixes = normalized @ self.hc_head_fn.T
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return (pre[..., None] * residual).sum(axis=2).astype(x.dtype)
