from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor as ImageProcessor
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, PILImageResampling


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class BaseImageProcessor(ImageProcessor):
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = (
            crop_size if crop_size is not None else {"height": 384, "width": 384}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    @abstractmethod
    def preprocess(self, images):
        pass


class KVCache:

    def __init__(self, head_dim, n_kv_heads, step=256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self, keys, values):
        self.update(keys, values)
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def fetch(self):
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def update(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (1, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (1, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

    @property
    def state(self):
        return self.keys, self.values


class SimpleKVCache:
    """A simple key-value cache for transformer attention layers.

    Stores and concatenates key/value tensors along sequence dimension.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.cache_length = 0

    def update_and_fetch(self, keys, values):
        """Update cache with new key/value tensors and return full cache.

        Args:
            keys: New key tensor to add [batch, heads, seq_len, head_dim]
            values: New value tensor to add [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cached_keys, cached_values) containing full cache history
        """
        if self.cache_length == 0:
            # First update - just store tensors
            self.keys = keys
            self.values = values
        else:
            # Concatenate with existing cache along sequence dimension
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.cache_length += keys.shape[2]
        return self.keys, self.values

    def fetch(self):
        return self.keys, self.values

    def update(self, keys, values):
        """Update cache with new key/value tensors without returning.

        Args:
            keys: New key tensor to store
            values: New value tensor to store
        """
        self.keys = keys
        self.values = values
        self.cache_length += keys.shape[2]


class RotatingKVCache:

    def __init__(self, head_dim, n_kv_heads, max_size, keep=None, step=256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep if keep is not None else step // 2
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def fetch(self):
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def update_and_fetch(self, keys, values):
        prev = self.offset
        B, _, S = keys.shape[:3]

        # Prefill mode
        if S > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                # The largest size is self.max_size + S - 1 to ensure
                # every token gets at least self.max_size context
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += S
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        # Generation mode
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + 1, :] = keys
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Optional[Any] = None):
    T = h.shape[1]
    if T > 1:
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if isinstance(c, RotatingKVCache):
                offset = min(c.max_size - 1, c.offset)
            else:
                offset = c.offset
        else:
            offset = 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask


@dataclass
class LanguageModelOutput:
    logits: mx.array
    cross_attention_states: Optional[List[mx.array]] = None
    encoder_outputs: Optional[List[mx.array]] = None


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = None
        self.language_model = None

    def filter_topk_vision_tokens(self, image_feature, attn, vision_filter_ratio=None):
        batch_size, seq_len = image_feature.shape[:2]
        k_tokens = (
            int(image_feature.shape[1] * vision_filter_ratio)
            if vision_filter_ratio is not None
            else None
        )  # keep 25% of the visual tokens

        if k_tokens is None or k_tokens == seq_len:
            return image_feature

        cls_idx = 0  # self.config.image_token_index

        attn_rec = mx.sum(attn[:, :, cls_idx + 1 :, cls_idx], axis=1)

        topk_idx = mx.argsort(attn_rec, axis=1)[:, -k_tokens:]

        # Create CLS token indices array
        # Shape: (B, 1)
        cls_indices = mx.full((batch_size, 1), cls_idx, dtype=mx.int32)

        # Concat with CLS token index
        # Add 1 to account for the offset after CLS token
        dominant_idx = mx.concatenate([cls_indices, topk_idx + cls_idx + 1], axis=1)

        image_feature = mx.take(image_feature, dominant_idx, axis=1)[0]
        return image_feature, dominant_idx

    def merge_similar_vision_tokens(
        self, image_feature, vision_merge_ratio, merge_rate=0.4
    ):
        # Skip CLS token (first token)
        tokens = image_feature[:, 1:]
        batch_size, num_tokens, hidden_dim = tokens.shape

        # Calculate target number of tokens
        target_tokens = max(1, int(num_tokens * vision_merge_ratio))

        # Create a mask of the same shape as tokens, initialized to True
        mask = mx.ones((batch_size, num_tokens))

        while num_tokens > target_tokens:
            # Calculate similarities between adjacent tokens
            tokens_a = tokens[:, :-1]  # all except last
            tokens_b = tokens[:, 1:]  # all except first

            # Calculate cosine similarity
            a_norm = mx.sqrt(mx.sum(tokens_a * tokens_a, axis=-1, keepdims=True))
            b_norm = mx.sqrt(mx.sum(tokens_b * tokens_b, axis=-1, keepdims=True))
            similarities = mx.sum(tokens_a * tokens_b, axis=-1)
            similarities = similarities / (a_norm.squeeze(-1) * b_norm.squeeze(-1))

            # Sort similarities and get indices of pairs to merge
            # We'll merge about 50% of remaining excess tokens in each iteration
            num_to_merge = max(1, int((num_tokens - target_tokens) * merge_rate))
            merge_indices = mx.argsort(similarities, axis=-1)[:, -num_to_merge:]

            # Create a list to track which indices to merge
            to_merge = set(merge_indices[0].tolist())

            # Merge selected pairs
            new_tokens = []
            new_mask = []
            i = 0
            while i < num_tokens:
                if i < num_tokens - 1 and i in to_merge:
                    # Merge this token with the next one
                    merged = (tokens[:, i : i + 1] + tokens[:, i + 1 : i + 2]) / 2
                    new_tokens.append(merged)
                    new_mask.append(mask[:, i : i + 1])  # Keep mask from first token
                    i += 2
                elif i > 0 and (i - 1) in to_merge:
                    # Skip this token as it was merged in the previous step
                    i += 1
                else:
                    # Keep this token as is
                    new_tokens.append(tokens[:, i : i + 1])
                    new_mask.append(mask[:, i : i + 1])
                    i += 1

            # Update tokens and mask
            tokens = mx.concatenate(new_tokens, axis=1)
            mask = mx.concatenate(new_mask, axis=1)
            num_tokens = tokens.shape[1]

        # Add back CLS token
        cls_mask = mx.ones((batch_size, 1), dtype=mx.bool_)
        return mx.concatenate([image_feature[:, :1], tokens], axis=1), mx.concatenate(
            [cls_mask, mask], axis=1
        )

    def merge_vision_patches(self, image_feature, vision_merge_ratio, merge_rate=0.4):
        pass
