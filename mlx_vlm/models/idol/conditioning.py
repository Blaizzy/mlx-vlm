"""Native MLX DINOv3 and FLUX.2 VAE image conditioning for Idol."""

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from ..flux2.vae.common.batch_norm_stats import Flux2BatchNormStats
from ..flux2.vae.encoder.encoder import Flux2Encoder
from ..sam3d_body.rope import apply_rope
from .weights import load_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config["hidden_size"]
        p = config.get("patch_size", 16)
        self.cls_token = mx.zeros((1, 1, c))
        self.mask_token = mx.zeros((1, 1, c))
        self.register_tokens = mx.zeros((1, config.get("num_register_tokens", 4), c))
        self.patch_embeddings = nn.Conv2d(config.get("num_channels", 3), c, p, stride=p)

    def __call__(self, x):
        h = self.patch_embeddings(x)
        b, _, _, c = h.shape
        return mx.concatenate(
            [
                mx.broadcast_to(self.cls_token, (b, 1, c)),
                mx.broadcast_to(
                    self.register_tokens, (b, self.register_tokens.shape[1], c)
                ),
                h.reshape(b, -1, c),
            ],
            axis=1,
        )


class DinoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        for name, bias in (
            ("q_proj", "query_bias"),
            ("k_proj", "key_bias"),
            ("v_proj", "value_bias"),
            ("o_proj", "proj_bias"),
        ):
            setattr(
                self, name, nn.Linear(c, c, bias=config.get(bias, bias != "key_bias"))
            )

    def __call__(self, x, rope):
        b, n, c = x.shape
        q, k, v = [
            getattr(self, name)(x).reshape(b, n, self.heads, -1).transpose(0, 2, 1, 3)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        q, k = apply_rope(q, k, *rope, n - rope[0].shape[0])
        h = mx.fast.scaled_dot_product_attention(
            q.astype(x.dtype), k.astype(x.dtype), v, scale=(c // self.heads) ** -0.5
        )
        return self.o_proj(h.transpose(0, 2, 1, 3).reshape(b, n, c))


class DinoMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        c, h = config["hidden_size"], config["intermediate_size"]
        bias = config.get("mlp_bias", True)
        self.gated = config.get("use_gated_mlp", False)
        self.activation = config.get("hidden_act", "gelu")
        if self.activation not in ("silu", "gelu"):
            raise ValueError(f"Unsupported DINOv3 activation: {self.activation}")
        self.up_proj = nn.Linear(c, h, bias=bias)
        self.down_proj = nn.Linear(h, c, bias=bias)
        if self.gated:
            self.gate_proj = nn.Linear(c, h, bias=bias)

    def __call__(self, x):
        act = nn.silu if self.activation == "silu" else nn.gelu
        h = (
            act(self.gate_proj(x)) * self.up_proj(x)
            if self.gated
            else act(self.up_proj(x))
        )
        return self.down_proj(h)


class DinoLayerScale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lambda1 = mx.ones((channels,))

    def __call__(self, x):
        return x * self.lambda1


class DinoLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config["hidden_size"]
        self.norm1 = nn.LayerNorm(c, eps=config.get("layer_norm_eps", 1e-5))
        self.norm2 = nn.LayerNorm(c, eps=config.get("layer_norm_eps", 1e-5))
        self.attention = DinoAttention(config)
        self.mlp = DinoMLP(config)
        self.layer_scale1 = DinoLayerScale(c)
        self.layer_scale2 = DinoLayerScale(c)

    def __call__(self, x, rope):
        x = x + self.layer_scale1(self.attention(self.norm1(x), rope))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DinoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = [DinoLayer(config) for _ in range(config["num_hidden_layers"])]

    def __call__(self, x, rope):
        for layer in self.layer:
            x = layer(x, rope)
        return x


class DINOv3(nn.Module):
    """HF DINOv3 ViT layout, including CLS and register tokens."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = DinoEmbeddings(config)
        self.encoder = DinoEncoder(config)
        self.norm = nn.LayerNorm(
            config["hidden_size"], eps=config.get("layer_norm_eps", 1e-5)
        )

    @staticmethod
    def sanitize(weights):
        # Transformers has used both top-level layer.* and model.layer.*.
        return {
            (
                "encoder." + key
                if key.startswith("layer.")
                else (
                    "encoder." + key[len("model.") :]
                    if key.startswith("model.layer.")
                    else key
                )
            ): value
            for key, value in weights.items()
        }

    def __call__(self, images):
        patch = self.config.get("patch_size", 16)
        ph, pw = (patch, patch) if isinstance(patch, int) else patch
        h, w = images.shape[1] // ph, images.shape[2] // pw
        y, x = mx.meshgrid(
            (mx.arange(h) + 0.5) / h * 2 - 1,
            (mx.arange(w) + 0.5) / w * 2 - 1,
            indexing="ij",
        )
        coords = mx.stack([y, x], axis=-1).reshape(-1, 2)
        d = self.config["hidden_size"] // self.config["num_attention_heads"]
        inv = self.config.get("rope_theta", 100.0) ** (
            -mx.arange(d // 4, dtype=mx.float32) * 4 / d
        )
        angles = (2 * math.pi * coords[..., None] * inv).reshape(h * w, -1)
        angles = mx.concatenate([angles, angles], axis=-1)
        rope = mx.sin(angles).astype(images.dtype), mx.cos(angles).astype(images.dtype)
        return self.norm(self.encoder(self.embeddings(images), rope))


class Flux2ImageEncoder(nn.Module):
    """Reuse the FLUX.2 encoder, sampling its posterior as Idol requires."""

    def __init__(self, config):
        super().__init__()
        c = config.get("latent_channels", 32)
        if c != 32 or config.get("patch_size", [2, 2]) not in (2, [2, 2], (2, 2)):
            raise ValueError(
                "Idol requires a FLUX.2 VAE with 32 latent channels and 2x2 packing"
            )
        self.encoder = Flux2Encoder(
            block_out_channels=tuple(
                config.get("block_out_channels", (128, 256, 512, 512))
            ),
            layers_per_block=config.get("layers_per_block", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            mid_block_add_attention=config.get("mid_block_add_attention", True),
        )
        self.quant_conv = (
            nn.Conv2d(2 * c, 2 * c, 1) if config.get("use_quant_conv", True) else None
        )
        self.bn = Flux2BatchNormStats(4 * c, eps=config.get("batch_norm_eps", 1e-4))

    def moments(self, images):
        h = self.encoder(images.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        mean, logvar = mx.split(h, 2, axis=-1)
        return mean, mx.clip(logvar, -30, 20)

    def __call__(self, images, noise=None):
        mean, logvar = self.moments(images)
        if noise is None:
            noise = mx.random.normal(mean.shape, dtype=mean.dtype)
        latent = mean + mx.exp(0.5 * logvar) * noise
        b, h, w, c = latent.shape
        # Channel order matches PyTorch (C, patch_y, patch_x).
        latent = (
            latent.reshape(b, h // 2, 2, w // 2, 2, c)
            .transpose(0, 1, 3, 5, 2, 4)
            .reshape(b, h // 2, w // 2, 4 * c)
        )
        mean = self.bn.running_mean.astype(latent.dtype)
        std = mx.sqrt(self.bn.running_var + self.bn.eps).astype(latent.dtype)
        tokens = ((latent - mean) / std).astype(mx.float32).reshape(b, -1, 4 * c)
        return mx.concatenate([mx.zeros((b, 5, 4 * c)), tokens], axis=1)


def resolve_encoder(repo_or_dir, *, vae=False, local_files_only=False):
    path = Path(repo_or_dir).expanduser()
    if not path.is_dir():
        patterns = (
            ["vae/config.json", "vae/*.safetensors"]
            if vae
            else ["config.json", "*.safetensors"]
        )
        path = Path(
            snapshot_download(
                str(repo_or_dir),
                allow_patterns=patterns,
                local_files_only=local_files_only,
            )
        )
    return path / "vae" if vae and (path / "vae").is_dir() else path


def read_encoder_weights(path):
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(file)))
    if not weights:
        raise FileNotFoundError(f"No safetensors weights in {path}")
    return weights


class ImageEncoder:
    def __init__(
        self, dinov3, flux2_vae, *, dinov3_dtype=mx.bfloat16, local_files_only=False
    ):
        dino_path = resolve_encoder(dinov3, local_files_only=local_files_only)
        vae_path = resolve_encoder(
            flux2_vae, vae=True, local_files_only=local_files_only
        )
        self.dinov3 = load_weights(
            DINOv3(json.loads((dino_path / "config.json").read_text())),
            read_encoder_weights(dino_path),
            dtype=dinov3_dtype,
        )
        vae_weights = {
            key.replace(".to_out.0.", ".to_out."): value
            for key, value in read_encoder_weights(vae_path).items()
            if key.startswith(("encoder.", "quant_conv.", "bn."))
        }
        self.flux2_vae = load_weights(
            Flux2ImageEncoder(json.loads((vae_path / "config.json").read_text())),
            vae_weights,
            dtype=mx.bfloat16,
        )

    def __call__(self, images):
        normalized = (images - mx.array(IMAGENET_MEAN)) / mx.array(IMAGENET_STD)
        dtype = self.dinov3.embeddings.patch_embeddings.weight.dtype
        tokens = self.dinov3(normalized.astype(dtype)).astype(mx.float32)
        feature1 = mx.fast.layer_norm(tokens, None, None, eps=1e-5)
        mx.eval(feature1)
        feature2 = self.flux2_vae(images.astype(mx.bfloat16) * 2 - 1)
        mx.eval(feature2)
        if feature1.shape[:2] != feature2.shape[:2]:
            raise ValueError(
                f"Encoder token counts differ: {feature1.shape}, {feature2.shape}"
            )
        return {"feature1": feature1, "feature2": feature2}
