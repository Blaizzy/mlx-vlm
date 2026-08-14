import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x: mx.array, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: mx.array, window_size: int, pad_hw: tuple, hw: tuple):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // ((Hp * Wp) // (window_size * window_size))
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        raise NotImplementedError(
            "Dynamic resolution rel_pos interpolation not yet supported in MLX port."
        )
    else:
        rel_pos_resized = rel_pos

    q_coords = mx.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = mx.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.astype(mx.int32)]


def add_decomposed_rel_pos(
    attn: mx.array,
    q: mx.array,
    rel_pos_h: mx.array,
    rel_pos_w: mx.array,
    q_size: tuple,
    k_size: tuple,
):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    # einsum, not broadcast-then-sum: the latter materialises 201M elements
    # per global block.
    rel_h = mx.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = mx.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.reshape(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    )
    return attn.reshape(B, q_h * q_w, k_h * k_w)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: tuple = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q = q.reshape(B * self.num_heads, H * W, -1)
        k = k.reshape(B * self.num_heads, H * W, -1)
        v = v.reshape(B * self.num_heads, H * W, -1)

        attn = (q * self.scale) @ k.transpose(0, 2, 1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).reshape(B, self.num_heads, H, W, -1)
        out = out.transpose(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        return self.proj(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: tuple = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.img_size = config.img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )

        if config.use_abs_pos:
            grid_size = config.img_size // config.patch_size
            self.pos_embed = mx.zeros((1, grid_size, grid_size, config.embed_dim))
        else:
            self.pos_embed = None

        self.blocks = []
        for i in range(config.depth):
            block = Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                use_rel_pos=config.use_rel_pos,
                window_size=(
                    config.window_size if i not in config.global_attn_indexes else 0
                ),
                input_size=(
                    config.img_size // config.patch_size,
                    config.img_size // config.patch_size,
                ),
            )
            self.blocks.append(block)

        self.conv1 = nn.Conv2d(
            config.embed_dim, config.out_chans, kernel_size=1, bias=False
        )
        self.norm1 = nn.LayerNorm(config.out_chans, eps=1e-6)
        self.conv2 = nn.Conv2d(
            config.out_chans, config.out_chans, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.LayerNorm(config.out_chans, eps=1e-6)

        # Two stride-2 convs: 64x64 neck output down to the 16x16 patch grid.
        self.net_2 = nn.Conv2d(
            config.out_chans,
            config.out_chans * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.net_3 = nn.Conv2d(
            config.out_chans * 2,
            config.out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x = self.net_2(x)
        x = self.net_3(x)

        # Output in PyTorch is B, C, H, W.
        # In MLX it is B, H, W, C.
        # modeling_GOT.py does: image_features.flatten(2).transpose(1, 2)
        # Which transforms B, C, H, W to B, H*W, C.
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)

        return x
