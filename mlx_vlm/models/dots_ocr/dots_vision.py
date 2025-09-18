import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., dim]
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int = 1536, hidden: int = 4224):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)  # up
        self.fc3 = nn.Linear(in_features, hidden, bias=False)  # gate
        self.fc2 = nn.Linear(hidden, in_features, bias=False)  # down

    def __call__(self, x: mx.array) -> mx.array:
        up = self.fc1(x)
        gate = self.fc3(x)
        sigmoid = 1.0 / (1.0 + mx.exp(-up))
        silu = up * sigmoid
        return self.fc2(silu * gate)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=1536, patch=14, eps=1e-6):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch, stride=patch, bias=False
        )
        self.norm = RMSNorm(embed_dim, eps=eps)
        self.patch = patch

    def __call__(self, x: mx.array):
        """
        x: [B,3,H,W]
        returns: tokens [B*Hp*Wp, embed_dim], Hp, Wp
        """
        x_nhwc = x.transpose(0, 2, 3, 1)  # [B, H, W, C]
        y = self.proj(x_nhwc)  # [B, Hp, Wp, D]
        B, Hp, Wp, D = y.shape
        y = y.reshape(B * Hp * Wp, D)
        y = self.norm(y)
        return y, int(Hp), int(Wp)
