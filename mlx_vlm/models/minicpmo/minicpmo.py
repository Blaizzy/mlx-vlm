from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..qwen3_vl.language import LanguageModel
from .audio import AudioModel, AudioProjector, check_conv1d_weight_shape
from .config import ModelConfig
from .processing_minicpmo import MiniCPMOProcessor  # noqa: F401
from .vision import VisionModel, check_array_shape


def _to_mx_array(value, dtype: Optional[mx.Dtype] = None):
    if value is None:
        return None
    if isinstance(value, mx.array):
        out = value
    elif hasattr(value, "detach"):
        out = mx.array(value.detach().cpu().numpy())
    elif hasattr(value, "numpy"):
        out = mx.array(value.numpy())
    else:
        out = mx.array(value)
    if dtype is not None and out.dtype != dtype:
        out = out.astype(dtype)
    return out


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, mx.array):
        return np.array(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.array(value)


def get_2d_sincos_pos_embed(image_size, embed_dim):
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=-1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    out = np.einsum("hw,d->hwd", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=-1)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        key_padding_mask: Optional[mx.array] = None,
    ):
        bsz, q_len, dim = queries.shape
        _, kv_len, _ = keys.shape

        q = self.q_proj(queries).reshape(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(keys).reshape(bsz, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(values).reshape(bsz, kv_len, self.num_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        mask = None
        if key_padding_mask is not None:
            mask = mx.where(
                key_padding_mask[:, None, None, :],
                mx.array(-1e9, dtype=q.dtype),
                mx.array(0.0, dtype=q.dtype),
            )

        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(bsz, q_len, dim)
        return self.out_proj(out)


class Resampler(nn.Module):
    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_size = max_size

        self.query = mx.zeros((self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = CrossAttention(embed_dim, num_heads)
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = mx.random.normal(shape=(embed_dim, embed_dim)) * (embed_dim**-0.5)

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size):
        pos_embed = get_2d_sincos_pos_embed(max_size, self.embed_dim)
        # Keep as numpy cache so it is not tracked as a model parameter.
        self.pos_embed = pos_embed.astype(np.float32)

    def _adjust_pos_cache(self, tgt_sizes):
        max_h = int(mx.max(tgt_sizes[:, 0]).item())
        max_w = int(mx.max(tgt_sizes[:, 1]).item())
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (max(max_h, self.max_size[0]), max(max_w, self.max_size[1]))
            self._set_2d_pos_cache(self.max_size)

    def __call__(self, x: mx.array, tgt_sizes: mx.array):
        assert x.shape[0] == tgt_sizes.shape[0]

        batch_size = x.shape[0]
        dtype = x.dtype
        patch_lens = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).astype(mx.int32)
        max_patch_len = int(mx.max(patch_lens).item())

        self._adjust_pos_cache(tgt_sizes)

        key_padding_mask = np.zeros((batch_size, max_patch_len), dtype=np.bool_)
        pos_embeds = []
        for i in range(batch_size):
            tgt_h, tgt_w = int(tgt_sizes[i, 0].item()), int(tgt_sizes[i, 1].item())
            pos = (
                mx.array(self.pos_embed[:tgt_h, :tgt_w, :])
                .reshape(tgt_h * tgt_w, -1)
                .astype(dtype)
            )

            cur_len = int(patch_lens[i].item())
            if cur_len < max_patch_len:
                pad = mx.zeros((max_patch_len - cur_len, pos.shape[-1]), dtype=dtype)
                pos = mx.concatenate([pos, pad], axis=0)
                key_padding_mask[i, cur_len:] = True
            pos_embeds.append(pos)

        pos_embeds = mx.stack(pos_embeds, axis=0)

        x = self.kv_proj(x)
        x = self.ln_kv(x)

        q = self.ln_q(self.query)
        q = mx.broadcast_to(
            q[None, :, :], (batch_size, self.num_queries, self.embed_dim)
        )

        out = self.attn(
            q,
            x + pos_embeds,
            x,
            key_padding_mask=mx.array(key_padding_mask),
        )
        out = self.ln_post(out)
        out = out @ self.proj
        return out


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.language_model = LanguageModel(config.text_config, config)
        self.vision_tower = VisionModel(config.vision_config)
        self.resampler = Resampler(
            num_queries=config.query_num,
            embed_dim=config.text_config.hidden_size,
            num_heads=max(1, config.text_config.hidden_size // 128),
            kv_dim=config.vision_config.hidden_size,
        )
        if config.init_audio and config.audio_config is not None:
            self.audio_tower = AudioModel(config.audio_config)
            audio_output_dim = int(config.audio_config.encoder_ffn_dim // 4)
            self.audio_projection_layer = AudioProjector(
                in_dim=audio_output_dim,
                out_dim=config.text_config.hidden_size,
            )
        else:
            self.audio_tower = None
            self.audio_projection_layer = None

    @property
    def layers(self):
        return self.language_model.model.layers

    def _set_position_state(self, input_ids: mx.array):
        batch_size, seq_len = input_ids.shape
        position_ids = mx.arange(seq_len, dtype=mx.int32).reshape(1, 1, -1)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = mx.zeros((batch_size, 1), dtype=mx.int32)

    def get_vision_embedding(self, pixel_values, tgt_sizes):
        dtype = self.language_model.model.embed_tokens.weight.dtype

        if pixel_values is None:
            return []

        batch_size = len(pixel_values)
        vision_hidden_states = []

        for batch_idx in range(batch_size):
            batch_pixels = (
                pixel_values[batch_idx] if batch_idx < len(pixel_values) else []
            )
            batch_tgt = tgt_sizes[batch_idx] if tgt_sizes is not None else []
            batch_tgt = _to_numpy(batch_tgt)
            if batch_tgt is None or len(batch_tgt) == 0:
                batch_tgt = np.zeros((0, 2), dtype=np.int32)
            else:
                batch_tgt = np.asarray(batch_tgt, dtype=np.int32)

            sample_embeddings = []
            for image_idx, cur_pixels in enumerate(batch_pixels):
                cur_pixels = _to_mx_array(cur_pixels, dtype=dtype)
                if cur_pixels is None:
                    continue
                if cur_pixels.ndim != 3:
                    continue

                # MiniCPM processor outputs tensors in C,H,W-like layout.
                if cur_pixels.shape[0] == 3:
                    cur_pixels = cur_pixels.transpose(1, 2, 0)
                cur_pixels = mx.expand_dims(cur_pixels, axis=0)

                if image_idx < len(batch_tgt):
                    cur_tgt = batch_tgt[image_idx]
                else:
                    seq_len = max(int(cur_pixels.shape[2] // self.config.patch_size), 1)
                    cur_tgt = np.array([1, seq_len], dtype=np.int32)

                cur_tgt = mx.array(cur_tgt, dtype=mx.int32)[None, :]
                patch_len = int((cur_tgt[0, 0] * cur_tgt[0, 1]).item())
                patch_attention_mask = mx.ones((1, 1, patch_len), dtype=mx.bool_)

                hidden = self.vision_tower(
                    cur_pixels,
                    patch_attention_mask=patch_attention_mask,
                    tgt_sizes=cur_tgt,
                )
                hidden = self.resampler(hidden, cur_tgt)
                sample_embeddings.append(hidden[0])

            if len(sample_embeddings) > 0:
                vision_hidden_states.append(mx.stack(sample_embeddings, axis=0))
            else:
                vision_hidden_states.append([])

        return vision_hidden_states

    def get_audio_embedding(self, audio_features, audio_feature_lens):
        if self.audio_tower is None or self.audio_projection_layer is None:
            return []
        if audio_features is None:
            return []

        audio_features = _to_mx_array(audio_features)
        if audio_features is None or audio_features.size == 0:
            return []

        if not isinstance(audio_feature_lens, list):
            audio_feature_lens = _to_numpy(audio_feature_lens)
            if audio_feature_lens is None:
                return []
            audio_feature_lens = [list(np.asarray(audio_feature_lens).tolist())]

        flat_lens = []
        for sample_lens in audio_feature_lens:
            if sample_lens is None:
                continue
            for v in sample_lens:
                flat_lens.append(int(v))

        if len(flat_lens) == 0:
            return [[] for _ in range(len(audio_feature_lens))]

        flat_lens_mx = mx.array(flat_lens, dtype=mx.int32)
        audio_states = self.audio_tower(audio_features, feature_lengths=flat_lens_mx)
        audio_embeds = self.audio_projection_layer(audio_states)

        # Pool with kernel=stride=audio_pool_step, matching HF path.
        pool_step = max(int(self.config.audio_pool_step), 1)
        pooled_count = max((audio_embeds.shape[1] - pool_step) // pool_step + 1, 0)
        pooled = []
        for i in range(pooled_count):
            start = i * pool_step
            end = start + pool_step
            pooled.append(mx.mean(audio_embeds[:, start:end, :], axis=1))

        if len(pooled) == 0:
            return [[] for _ in range(len(audio_feature_lens))]
        pooled = mx.stack(pooled, axis=1)

        feature_lens_after_cnn = ((flat_lens_mx - 1) // 2) + 1
        feature_lens_after_pool = (
            (feature_lens_after_cnn - pool_step) // pool_step
        ) + 1
        feature_lens_after_pool = mx.maximum(feature_lens_after_pool, 1)
        feature_lens_after_pool = _to_numpy(feature_lens_after_pool)

        outputs = []
        idx = 0
        for sample_lens in audio_feature_lens:
            sample_embeds = []
            for _ in sample_lens:
                cur_len = int(feature_lens_after_pool[idx])
                sample_embeds.append(pooled[idx, :cur_len, :])
                idx += 1
            outputs.append(sample_embeds)
        return outputs

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[list] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        self._set_position_state(input_ids)

        tgt_sizes = kwargs.get("tgt_sizes", None)
        image_bound = kwargs.get("image_bound", None)
        vision_hidden_states = (
            self.get_vision_embedding(pixel_values, tgt_sizes)
            if pixel_values is not None
            else None
        )

        audio_features = kwargs.get("audio_features", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)
        audio_bounds = kwargs.get("audio_bounds", None)
        audio_hidden_states = (
            self.get_audio_embedding(audio_features, audio_feature_lens)
            if audio_features is not None
            else []
        )

        updated = []
        for batch_idx in range(inputs_embeds.shape[0]):
            cur_embeds = inputs_embeds[batch_idx]

            # Vision embeddings replacement.
            if vision_hidden_states is not None and image_bound is not None:
                cur_vs_hs = vision_hidden_states[batch_idx]
                if isinstance(cur_vs_hs, mx.array) and cur_vs_hs.size > 0:
                    cur_bounds = (
                        image_bound[batch_idx]
                        if isinstance(image_bound, list)
                        else image_bound[batch_idx]
                    )
                    cur_bounds = _to_numpy(cur_bounds)
                    if cur_bounds is not None and cur_bounds.size > 0:
                        cur_bounds = np.asarray(cur_bounds, dtype=np.int32).reshape(
                            -1, 2
                        )
                        indices = []
                        for start, end in cur_bounds:
                            if end > start:
                                indices.append(np.arange(start, end, dtype=np.int32))

                        if len(indices) > 0:
                            indices = np.concatenate(indices, axis=0)
                            features = cur_vs_hs.reshape(
                                -1, cur_vs_hs.shape[-1]
                            ).astype(cur_embeds.dtype)
                            usable = min(features.shape[0], len(indices))
                            if usable > 0:
                                indices_mx = mx.array(indices[:usable], dtype=mx.uint32)
                                features = features[:usable]
                                current = mx.take(cur_embeds, indices_mx, axis=0)
                                cur_embeds = cur_embeds.at[indices_mx].add(
                                    features - current
                                )

            # Audio embeddings replacement.
            if (
                isinstance(audio_hidden_states, list)
                and batch_idx < len(audio_hidden_states)
                and audio_bounds is not None
            ):
                cur_audio_embeds = audio_hidden_states[batch_idx]
                cur_audio_bounds = (
                    audio_bounds[batch_idx]
                    if isinstance(audio_bounds, list)
                    else audio_bounds[batch_idx]
                )
                cur_audio_bounds = _to_numpy(cur_audio_bounds)
                if cur_audio_bounds is not None and len(cur_audio_embeds) > 0:
                    cur_audio_bounds = np.asarray(
                        cur_audio_bounds, dtype=np.int32
                    ).reshape(-1, 2)
                    for seg_idx, (start, end) in enumerate(cur_audio_bounds):
                        if seg_idx >= len(cur_audio_embeds) or end <= start:
                            continue

                        seg_features = cur_audio_embeds[seg_idx]
                        if not isinstance(seg_features, mx.array):
                            seg_features = _to_mx_array(
                                seg_features, dtype=cur_embeds.dtype
                            )
                        else:
                            seg_features = seg_features.astype(cur_embeds.dtype)
                        if seg_features is None or seg_features.size == 0:
                            continue

                        indices = np.arange(start, end, dtype=np.int32)
                        usable = min(seg_features.shape[0], len(indices))
                        if usable <= 0:
                            continue
                        indices_mx = mx.array(indices[:usable], dtype=mx.uint32)
                        seg_features = seg_features[:usable]
                        current = mx.take(cur_embeds, indices_mx, axis=0)
                        cur_embeds = cur_embeds.at[indices_mx].add(
                            seg_features - current
                        )

            updated.append(cur_embeds)

        inputs_embeds = mx.stack(updated, axis=0)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[list] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )
        return self.language_model(
            input_ids,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        in_proj_weight = None
        in_proj_bias = None

        for key, value in weights.items():
            if key.startswith(("tts.", "audio_avg_pooler.")):
                continue

            if key.startswith("llm."):
                key = key.replace("llm.", "language_model.", 1)
            elif key.startswith("vpm."):
                key = key.replace("vpm.", "vision_tower.", 1)
            elif key.startswith("apm."):
                key = key.replace("apm.", "audio_tower.", 1)
            elif key.startswith("audio_projection_layer."):
                pass
            elif key.startswith("resampler."):
                pass
            else:
                continue

            if key == "resampler.attn.in_proj_weight":
                in_proj_weight = value
                continue
            if key == "resampler.attn.in_proj_bias":
                in_proj_bias = value
                continue

            if "position_ids" in key:
                continue
            if key.endswith(
                "embeddings.patch_embedding.weight"
            ) and not check_array_shape(value):
                value = value.transpose(0, 2, 3, 1)
            if (
                key.endswith("audio_tower.conv1.weight")
                or key.endswith("audio_tower.conv2.weight")
            ) and not check_conv1d_weight_shape(value):
                value = value.transpose(0, 2, 1)

            sanitized_weights[key] = value

        if in_proj_weight is not None:
            q_w, k_w, v_w = mx.split(in_proj_weight, 3, axis=0)
            sanitized_weights["resampler.attn.q_proj.weight"] = q_w
            sanitized_weights["resampler.attn.k_proj.weight"] = k_w
            sanitized_weights["resampler.attn.v_proj.weight"] = v_w
        if in_proj_bias is not None:
            q_b, k_b, v_b = mx.split(in_proj_bias, 3, axis=0)
            sanitized_weights["resampler.attn.q_proj.bias"] = q_b
            sanitized_weights["resampler.attn.k_proj.bias"] = k_b
            sanitized_weights["resampler.attn.v_proj.bias"] = v_b

        if self.config.text_config.tie_word_embeddings:
            sanitized_weights.pop("language_model.lm_head.weight", None)

        return sanitized_weights
