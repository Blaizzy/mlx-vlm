"""Mage-ViT — the codec-native visual encoder of Mage-VL.

Isomorphic to `modeling_mage_vl.py` on the hub: same class names, same decomposition, same
forward order. Diff the two and you should see only PyTorch↔MLX op substitutions.

Three things in here are deliberate reproductions of upstream behaviour that look like bugs.
Do not "fix" them — the weights were trained with them:

1. `rotate_half` is INTERLEAVED, `(x1,x2,x3,x4) -> (-x2,x1,-x4,x3)`, not the usual split-half.
   Upstream's own docstring says "to match Source model's implementation."
2. The rope frequency table is `concat([freqs, freqs])`, so under an *interleaved* rotation the
   cos/sin at lanes 2i and 2i+1 come from DIFFERENT frequencies. Textbook RoPE pairs equal
   frequencies. This pairing is what the model was trained with; reproduce it exactly.
3. The tap is the LAST hidden state, not the second-to-last, despite upstream's comment saying
   "Use second-to-last layer output" — the code reads `hidden_states[-1]`. Follow the code.
   (An off-by-one on exactly this tap is what broke the Anima port; see the mlx-porting skill.)
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def rotate_half(x: mx.array) -> mx.array:
    """Interleaved rotation: (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)."""
    shape = x.shape
    pairs = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]
    out = mx.stack([-x_odd, x_even], axis=-1)
    return out.reshape(*shape)


def apply_rotary_pos_emb(
    q: mx.array, k: mx.array, freqs: mx.array
) -> Tuple[mx.array, mx.array]:
    """q, k: (B, H, L, D) · freqs: (B, L, D). Computed in fp32 like upstream."""
    orig_dtype = q.dtype
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    cos = mx.expand_dims(mx.cos(freqs), axis=1).astype(mx.float32)
    sin = mx.expand_dims(mx.sin(freqs), axis=1).astype(mx.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


def get_norm_layer(config: VisionConfig):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class VisionRotaryEmbedding(nn.Module):
    """3D (T,H,W) rotary frequency constructor with a 4:6:6 split of head_dim//2."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert (
            half % 16 == 0
        ), "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half
        self.base = base

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        # NOTE the exponent: arange(size)/size, NOT the conventional arange(0, d, 2)/d.
        self._inv_freq_t = 1.0 / (
            base ** (mx.arange(self.t_size, dtype=mx.float32) / self.t_size)
        )
        self._inv_freq_h = 1.0 / (
            base ** (mx.arange(self.h_size, dtype=mx.float32) / self.h_size)
        )
        self._inv_freq_w = 1.0 / (
            base ** (mx.arange(self.w_size, dtype=mx.float32) / self.w_size)
        )
        # Evaluate at init so the caches don't carry a lazy graph across streams.
        mx.eval(self._inv_freq_t, self._inv_freq_h, self._inv_freq_w)

    def forward_from_positions(self, patch_positions: mx.array) -> mx.array:
        """patch_positions: (L, 3) of [t, h, w] -> freqs (L, half)."""
        t_pos = patch_positions[:, 0].astype(mx.float32)
        h_pos = patch_positions[:, 1].astype(mx.float32)
        w_pos = patch_positions[:, 2].astype(mx.float32)

        ft = mx.outer(t_pos, self._inv_freq_t)
        fh = mx.outer(h_pos, self._inv_freq_h)
        fw = mx.outer(w_pos, self._inv_freq_w)
        return mx.concatenate([ft, fh, fw], axis=-1)

    def forward_with_thw(self, t: int, h: int, w: int) -> mx.array:
        """Grid form: patches ordered t-major, then h, then w."""
        t_ids = mx.repeat(mx.arange(t), h * w)
        h_ids = mx.tile(mx.repeat(mx.arange(h), w), [t])
        w_ids = mx.tile(mx.arange(w), [h * t])
        positions = mx.stack([t_ids, h_ids, w_ids], axis=-1)
        return self.forward_from_positions(positions)


class MageVLVisionEmbeddings(nn.Module):
    """Patch embedding over patches already arranged in 2x2 block order by the processor."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        p, c = self.patch_size, self.in_channels
        # Upstream views as (N, C, P, P); MLX convolutions are NHWC, so land in (N, P, P, C).
        x = hidden_states.reshape(-1, c, p, p).transpose(0, 2, 3, 1)
        x = self.patch_embedding(x)
        return x.reshape(-1, self.embed_dim)


class MageVLVisionPatchMerger(nn.Module):
    """Merge spatial_merge_size^2 patches into one and project to the LLM width."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-5,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=layer_norm_eps)
        # nn.Sequential in upstream -> indices 0 and 2 in the weight names.
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]
        self.spatial_merge_size = spatial_merge_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.patch_position_encoding_type = patch_position_encoding_type

        if self.use_patch_position_encoding:
            if self.patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unknown patch_position_encoding_type: {self.patch_position_encoding_type}."
                )
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def __call__(
        self, x: mx.array, patch_positions: Optional[mx.array] = None
    ) -> mx.array:
        if patch_positions is not None and patch_positions.ndim == 3:
            patch_positions = patch_positions.squeeze(0)

        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)

        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.reshape(-1, self.spatial_merge_size**2, 3)[:, 0, :]
            pp = pp // self.spatial_merge_size
            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])

        return x


class MageVLVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        # (B, L, 3*H*D) -> (B, L, 3, H, D) -> 3 x (B, H, L, D). Upstream reshapes then permutes
        # (2,0,1,3,4) then transposes(1,2); the QKV split is CONTIGUOUS per projection, not
        # head-interleaved — keep this exact order.
        qkv = self.qkv(hidden_states).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rotary_pos_emb is not None:
            q, k = apply_rotary_pos_emb(q, k, rotary_pos_emb)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.proj(out)


class SiglipMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        # config.hidden_act is "gelu": the EXACT erf gelu, not the tanh approximation.
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.activation_fn(self.fc1(x)))


class MageVLVisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MageVLVisionAttention(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, rotary_pos_emb)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MageVLVisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [
            MageVLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_pos_emb)
        return hidden_states


def build_cu_seqlens(
    grid_thw: Optional[List[Tuple[int, int, int]]],
    total_patches: int,
    fixed_t: int = 4,
) -> List[int]:
    """Cumulative window boundaries. Windows split a sample's frames into `fixed_t` chunks."""
    if not grid_thw:
        return [0, total_patches]

    cu = [0]
    current = 0
    for t, h, w in grid_thw:
        if fixed_t and fixed_t > 0 and t > fixed_t:
            for _ in range(t // fixed_t):
                current += fixed_t * h * w
                cu.append(current)
            if t % fixed_t:
                current += (t % fixed_t) * h * w
                cu.append(current)
        else:
            current += t * h * w
            cu.append(current)

    if cu[-1] != total_patches:
        raise ValueError(
            f"cu_seqlens mismatch: total_patches={total_patches} calculated={cu[-1]} grid={grid_thw}"
        )
    return cu


def block_diagonal_mask(cu_seqlens: List[int], dtype: mx.Dtype) -> Optional[mx.array]:
    """Additive mask confining attention to each window. None when there is a single block."""
    if len(cu_seqlens) <= 2:
        return None
    total = cu_seqlens[-1]
    ids = mx.zeros((total,), dtype=mx.int32)
    for i in range(len(cu_seqlens) - 1):
        lo, hi = cu_seqlens[i], cu_seqlens[i + 1]
        ids = ids + mx.where(
            (mx.arange(total) >= lo) & (mx.arange(total) < hi), mx.array(i, mx.int32), 0
        )
    same = mx.expand_dims(ids, 1) == mx.expand_dims(ids, 0)
    return mx.where(same, mx.array(0, dtype), mx.array(-mx.inf, dtype))[None, None]


class VisionModel(nn.Module):
    """Mage-ViT. `use_head` is False in the VLM, so the pooling head is absent by construction."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.embeddings = MageVLVisionEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = MageVLVisionEncoder(config)
        self.video_rope = VisionRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
        else:
            self.layernorm_post = None

        self.merger = MageVLVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            layer_norm_eps=config.layer_norm_eps,
            use_patch_position_encoding=config.use_patch_position_encoding,
            patch_position_encoding_type=config.patch_position_encoding_type,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_state: mx.array,
        patch_positions: mx.array,
        grid_thw: Optional[List[Tuple[int, int, int]]] = None,
        skip_merger: bool = False,
        cu_seqlens: Optional[List[int]] = None,
    ) -> mx.array:
        """``cu_seqlens`` supplies attention-window boundaries directly.

        The dense path derives them from ``grid_thw`` (``fixed_t * h * w`` per window),
        which cannot describe a codec-selected patch set: sparse selection keeps a
        different number of patches in every window. Callers doing sparse video pass the
        boundaries they actually built. Everything else in the tower is already
        sparsity-agnostic -- ``forward_from_positions`` takes arbitrary ``(L, 3)``
        coordinates, and 3D RoPE is defined over the un-pruned grid by design.
        """
        hidden_states = self.embeddings(hidden_state)
        if hidden_states.ndim == 2:
            hidden_states = mx.expand_dims(hidden_states, 0)
        _, total_patches, _ = hidden_states.shape

        if patch_positions.ndim == 3:
            patch_positions = patch_positions.squeeze(0)
        freqs = self.video_rope.forward_from_positions(patch_positions)
        # See module docstring, item 2: plain concat, NOT repeat_interleave.
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        if freqs.ndim == 2:
            freqs = mx.expand_dims(freqs, 0)

        hidden_states = self.layernorm_pre(hidden_states)

        if cu_seqlens is not None:
            cu = list(cu_seqlens)
            if not cu or cu[0] != 0 or cu[-1] != total_patches:
                raise ValueError(
                    f"cu_seqlens must start at 0 and end at total_patches={total_patches}; "
                    f"got {cu[:3]}...{cu[-1:]}"
                )
        else:
            cu = build_cu_seqlens(grid_thw, total_patches, self.config.frame_windows_size)
        mask = block_diagonal_mask(cu, hidden_states.dtype)

        # Item 3: upstream takes hidden_states[-1] — the final layer — despite its comment.
        sequence_output = self.encoder(hidden_states, mask, freqs)

        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        if skip_merger:
            return sequence_output

        return self.merger(sequence_output, patch_positions=patch_positions)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "patch_embedding.weight" in k and getattr(v, "ndim", 0) == 4:
                # PyTorch Conv2d (O, I, kH, kW) -> MLX (O, kH, kW, I). Guarded on the input-channel
                # axis so a re-sanitize of already-converted weights is a no-op.
                if v.shape[1] == self.config.num_channels:
                    v = v.transpose(0, 2, 3, 1)
            sanitized[k] = v
        return sanitized
