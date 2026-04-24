from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..qwen3_5.language import LanguageModel as Qwen35LanguageModel
from .config import ModelConfig
from .processing_minicpmv import MiniCPMVProcessor  # noqa: F401
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


class VitMerger(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        merged_hidden_size: int = 17216,
        num_heads: int = 16,
        merge_group_size: int = 2,
    ):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.merge_group_size = merge_group_size
        self.group_tokens = merge_group_size * merge_group_size
        self.group_hidden_size = vision_hidden_size * self.group_tokens

        self.pre_norm = nn.LayerNorm(self.group_hidden_size, eps=1e-6)
        self.self_attn = CrossAttention(vision_hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(self.group_hidden_size, merged_hidden_size, bias=True)
        self.linear_2 = nn.Linear(merged_hidden_size, vision_hidden_size, bias=True)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x: mx.array, grid_h: int, grid_w: int) -> tuple[mx.array, int, int]:
        group = self.merge_group_size
        if grid_h % group != 0 or grid_w % group != 0:
            raise ValueError(
                "MiniCPM-V vit_merger requires target grid divisible by 2, "
                f"got {(grid_h, grid_w)}."
            )

        hidden_dim = int(x.shape[-1])
        merged_h = grid_h // group
        merged_w = grid_w // group

        windows = x.reshape(grid_h, grid_w, hidden_dim)
        windows = windows.reshape(
            merged_h, group, merged_w, group, hidden_dim
        ).transpose(0, 2, 1, 3, 4)
        windows = windows.reshape(merged_h * merged_w, self.group_tokens, hidden_dim)

        normed_windows = self.layer_norm1(windows)
        attn_windows = self.self_attn(normed_windows, normed_windows, normed_windows)
        windows = windows + attn_windows

        residual = mx.mean(windows, axis=1)
        merged = windows.reshape(merged_h * merged_w, self.group_hidden_size)
        merged = self.pre_norm(merged)
        merged = self.linear_1(merged)
        merged = self.act(merged)
        merged = self.linear_2(merged)
        return merged + residual, merged_h, merged_w


class MergerBlock(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = [
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.GELU(approx="precise"),
            nn.Linear(hidden_size, out_size, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_norm(x)
        for layer in self.mlp:
            x = layer(x)
        return x


class Merger(nn.Module):
    def __init__(self, hidden_size: int, out_size: int, merger_times: int = 1):
        super().__init__()
        self.mlp = [
            MergerBlock(
                hidden_size * 4 if i == 0 else (hidden_size if i < merger_times else out_size) * 4,
                out_size if i == merger_times - 1 else hidden_size,
            )
            for i in range(merger_times)
        ]

    def __call__(self, x: mx.array, grid_h: int, grid_w: int) -> tuple[mx.array, int, int]:
        cur_h = int(grid_h)
        cur_w = int(grid_w)
        hidden = x

        for layer_idx, layer in enumerate(self.mlp):
            if cur_h % 2 != 0 or cur_w % 2 != 0:
                raise ValueError(
                    "MiniCPM-V merger requires target grid divisible by 2, "
                    f"got {(cur_h, cur_w)} at merge round {layer_idx}."
                )

            inner_dim = int(hidden.shape[-1])
            merged_h = cur_h // 2
            merged_w = cur_w // 2
            hidden = hidden.reshape(cur_h, cur_w, inner_dim)
            hidden = hidden.reshape(merged_h, 2, merged_w, 2, inner_dim)
            hidden = hidden.transpose(0, 2, 1, 3, 4)
            hidden = hidden.reshape(merged_h * merged_w, inner_dim * 4)
            hidden = layer(hidden)
            cur_h, cur_w = merged_h, merged_w

        return hidden, cur_h, cur_w


class LanguageModel(Qwen35LanguageModel):
    pass


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.language_model = LanguageModel(config.text_config, config)
        self.vision_tower = VisionModel(config.vision_config)
        self.vit_merger = VitMerger(
            vision_hidden_size=config.vision_config.hidden_size,
            merged_hidden_size=17216,
            num_heads=config.vision_config.num_attention_heads,
            merge_group_size=2,
        )
        self.merger = Merger(
            hidden_size=config.vision_config.hidden_size,
            out_size=config.text_config.hidden_size,
            merger_times=int(getattr(config, "merger_times", 1) or 1),
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def _set_position_state(self, input_ids: mx.array):
        batch_size, seq_len = input_ids.shape
        position_ids = mx.arange(seq_len, dtype=mx.int32).reshape(1, 1, -1)
        position_ids = mx.broadcast_to(position_ids, (3, batch_size, seq_len))
        self.language_model._position_ids = position_ids
        self.language_model._rope_deltas = mx.zeros((batch_size, 1), dtype=mx.int32)

    @staticmethod
    def _apply_vision_delta(target_states: mx.array, indices_mx: mx.array, features: mx.array) -> mx.array:
        current = mx.take(target_states, indices_mx, axis=0)
        return target_states.at[indices_mx].add(features - current)

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

                # The vision tower convolution path expects HWC inputs.
                # Convert both CHW and patch-packed (C, patch, n * patch) layouts to HWC here.
                if cur_pixels.shape[0] == 3:
                    cur_pixels = cur_pixels.transpose(1, 2, 0)
                cur_pixels = mx.expand_dims(cur_pixels, axis=0)

                if image_idx < len(batch_tgt):
                    cur_tgt = batch_tgt[image_idx]
                else:
                    seq_len = max(int(cur_pixels.shape[2] // self.config.patch_size), 1)
                    cur_tgt = np.array([1, seq_len], dtype=np.int32)

                cur_tgt = mx.array(cur_tgt, dtype=mx.int32)[None, :]
                hidden = self.vision_tower.embeddings(
                    cur_pixels,
                    tgt_sizes=cur_tgt,
                )
                hidden = hidden.astype(self.vision_tower.embeddings.patch_embedding.weight.dtype)
                grid_h = int(cur_tgt[0, 0].item())
                grid_w = int(cur_tgt[0, 1].item())

                insert_layer_id = int(getattr(self.config, "insert_layer_id", 6) or 6)
                use_vit_merger = str(getattr(self.config, "downsample_mode", "16x")) != "4x"
                for layer_index, encoder_layer in enumerate(self.vision_tower.encoder.layers):
                    hidden = encoder_layer(hidden, attention_mask=None)
                    if use_vit_merger and layer_index == insert_layer_id:
                        merged_hidden, grid_h, grid_w = self.vit_merger(hidden[0], grid_h, grid_w)
                        hidden = mx.expand_dims(merged_hidden, axis=0)

                hidden = self.vision_tower.post_layernorm(hidden)
                hidden = hidden[0]

                merged_tokens, final_h, final_w = self.merger(hidden, grid_h, grid_w)
                merged_tokens = merged_tokens.astype(dtype)
                sample_embeddings.append(merged_tokens)

            if len(sample_embeddings) > 0:
                vision_hidden_states.append(mx.concatenate(sample_embeddings, axis=0))
            else:
                vision_hidden_states.append([])

        return vision_hidden_states

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[list] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        tgt_sizes = kwargs.get("tgt_sizes", None)
        image_bound = kwargs.get("image_bound", None)
        self._set_position_state(input_ids)
        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            vision_hidden_states = cached
        elif pixel_values is not None:
            vision_hidden_states = self.get_vision_embedding(pixel_values, tgt_sizes)
        else:
            vision_hidden_states = None

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
                            if int(features.shape[0]) != int(len(indices)):
                                raise ValueError(
                                    "MiniCPM-V vision feature token count does not match "
                                    f"image placeholder span length for sample {batch_idx}: "
                                    f"features={features.shape[0]} placeholders={len(indices)}."
                                )
                            indices_mx = mx.array(indices, dtype=mx.uint32)
                            cur_embeds = self._apply_vision_delta(
                                cur_embeds, indices_mx, features
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
        output = self.language_model(
            input_ids,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
        )
        return output

    def sanitize(self, weights):
        sanitized_weights = {}
        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        for key, value in weights.items():
            # MiniCPM-V 4.6 checkpoint namespaces are prefixed with `model.`
            # and split across language_model / vpm / vit_merger / merger.
            if key.startswith("model."):
                key = key.replace("model.", "", 1)

            mapped_key = None
            if key.startswith("language_model."):
                mapped_key = key.replace("language_model.", "language_model.model.", 1)
            elif key.startswith("lm_head."):
                mapped_key = key.replace("lm_head.", "language_model.lm_head.", 1)
            elif key.startswith("vpm."):
                mapped_key = key.replace("vpm.", "vision_tower.", 1)
            elif key.startswith("vit_merger.") or key.startswith("merger."):
                mapped_key = key
            # Backward compatibility with older naming schemes.
            elif key.startswith("llm."):
                mapped_key = key.replace("llm.", "language_model.model.", 1)
            elif key.startswith("visual."):
                mapped_key = key.replace("visual.", "vision_tower.", 1)

            if mapped_key is None:
                continue
            key = mapped_key

            if "position_ids" in key:
                continue
            if "conv1d.weight" in key and len(value.shape) == 3 and value.shape[-1] != 1:
                value = value.transpose(0, 2, 1)
            if key.endswith(
                "embeddings.patch_embedding.weight"
            ) and not check_array_shape(value):
                value = value.transpose(0, 2, 3, 1)
            if any(key.endswith(suffix) for suffix in norm_keys) and value.ndim == 1:
                value = value + 1.0

            sanitized_weights[key] = value

        if self.config.text_config.tie_word_embeddings:
            sanitized_weights.pop("language_model.lm_head.weight", None)

        return sanitized_weights
