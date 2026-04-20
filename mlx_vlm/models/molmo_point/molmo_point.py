import math
import re
from copy import deepcopy
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import AdapterConfig, ModelConfig, VisionConfig
from .language import LanguageModel
from .vision import VisionModel, ViTMultiHeadDotProductAttention

EXTRACT_POINT_TRIPLE = re.compile(
    r"<POINT_(\d+)> ?<POINT_(\d+)> ?<POINT_(\d+)> ?([0-9]+)"
)


def get_subpatch_ids(output_text, pooling, no_more_points_class):
    n_patches, n_subpatches = pooling.shape[-2:]
    if no_more_points_class:
        n_patches += 1
    for match in EXTRACT_POINT_TRIPLE.finditer(output_text):
        patch_id, subpatch_num = int(match.group(1)), int(match.group(2))
        subpatch_id = subpatch_num - n_patches
        location_num = int(match.group(3))
        location_id = location_num - n_patches - n_subpatches
        example_id = int(match.group(4))
        vit_patch_id = pooling[patch_id, subpatch_id]
        yield vit_patch_id, location_id, example_id


def extract_image_points(
    output_text, pooling, mappings, no_more_points_class, location, image_sizes
):
    if len(mappings) != len(image_sizes):
        raise ValueError("Mapping and image sizes must have the same length")
    extracted_points = []
    for vit_patch_id, location_id, example_id in get_subpatch_ids(
        output_text, pooling, no_more_points_class
    ):
        for image_ix, (mapping, (w, h)) in enumerate(zip(mappings, image_sizes)):
            patch_coords = np.argwhere(mapping == int(vit_patch_id))
            if len(patch_coords) == 1:
                p_y, p_x = patch_coords[0]
                if location_id is not None:
                    loc_x = location_id // 3
                    loc_y = location_id % 3
                    p_x += (loc_x + 0.5) * 0.33
                    p_y += (loc_y + 0.5) * 0.33
                else:
                    p_x += 0.5
                    p_y += 0.5
                extracted_points.append(
                    [
                        example_id,
                        image_ix,
                        (p_x / mapping.shape[1]) * w,
                        (p_y / mapping.shape[0]) * h,
                    ]
                )
                break
    return extracted_points


class ImageProjectorMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, hidden_act: str
    ):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class AddPosEmbed(nn.Module):
    def __init__(self, in_features: int, n_pos: int):
        super().__init__()
        self.bias = mx.zeros((n_pos, in_features))

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.bias[None, : x.shape[-2], :]


class MolmoPointPadWithLearnedVector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.vector = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        vector = mx.broadcast_to(
            self.vector[None, None, :], (B, 1, self.vector.shape[0])
        )
        return mx.concatenate([x, vector], axis=1)


class MolmoPointPatchRope(nn.Module):
    def __init__(self, theta: float, dim: int):
        super().__init__()
        self._inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def rotate_half(self, x: mx.array) -> mx.array:
        B, hs = x.shape
        x = x.reshape(B, 2, hs // 2)
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        return mx.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: mx.array, position_ids: mx.array) -> mx.array:
        inv_freq = self._inv_freq.astype(mx.float32)
        position_ids = position_ids.astype(mx.float32)
        x_float = x.astype(mx.float32)
        freqs = position_ids[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        out = (x_float * cos) + (self.rotate_half(x_float) * sin)
        return out.astype(x.dtype)


class MolmoPointConnector(nn.Module):
    def __init__(self, config: AdapterConfig, vit_config: VisionConfig):
        super().__init__()
        self.config = config
        n_vit_layers = len(config.vit_layers)
        pool_dim = vit_config.hidden_size * n_vit_layers

        self.image_projector = ImageProjectorMLP(
            config.hidden_size,
            config.intermediate_size,
            config.text_hidden_size,
            config.hidden_act,
        )
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            input_dim=pool_dim,
            out_layer=False,
        )
        if config.positional_embeddings:
            self.positional_embeddings = AddPosEmbed(
                pool_dim, config.positional_embeddings
            )
        else:
            self.positional_embeddings = None

    def __call__(self, to_pool: mx.array, to_pool_mask: mx.array) -> mx.array:
        if self.positional_embeddings is not None:
            to_pool = self.positional_embeddings(to_pool)

        if self.config.pooling_attention_mask:
            attn_mask = to_pool_mask.reshape(-1, 1, 1, to_pool_mask.shape[-1])
        else:
            attn_mask = None
            to_pool = to_pool * to_pool_mask.astype(to_pool.dtype)[:, :, None]

        denom = to_pool_mask.reshape(-1, to_pool.shape[-2]).astype(mx.float32).sum(-1)
        denom = mx.where(denom == 0, 1, denom)
        query = to_pool.sum(-2, keepdims=True) / denom[:, None, None]

        pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled_features = self.image_projector(pooled_features)
        return pooled_features


class PointPredictor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        llm_dim = config.text_config.hidden_size
        patch_embed_dim = config.patch_embed_dim
        vit_dim = config.vision_config.hidden_size * len(
            config.adapter_config.vit_layers
        )

        if config.layer_norm_x:
            self.x_norm = nn.RMSNorm(llm_dim, eps=config.text_config.layer_norm_eps)
        else:
            self.x_norm = None

        if config.token_prediction_rotary == "one_d":
            theta = (
                config.token_prediction_rotary_theta or config.text_config.rope_theta
            )
            self.patch_rotary = MolmoPointPatchRope(theta, patch_embed_dim)
        else:
            self.patch_rotary = None

        self.patch_q = nn.Linear(llm_dim, patch_embed_dim)
        self.patch_k = nn.Linear(llm_dim, patch_embed_dim)
        self.subpatch_q = nn.Linear(llm_dim, patch_embed_dim)
        self.subpatch_k = nn.Linear(vit_dim, patch_embed_dim)
        self.add_no_point_class_embed = MolmoPointPadWithLearnedVector(patch_embed_dim)

        if config.patch_location == "3x3":
            self.subpatch_loc_k = nn.Linear(llm_dim, 9)
        else:
            self.subpatch_loc_k = None


class GeneratedTokenBounds:
    def __init__(
        self, vocab_size, n_patches, n_subpatches, n_locations, no_more_points_class
    ):
        self.n_locations = n_locations
        self.n_patches = n_patches
        self.n_subpatches = n_subpatches
        self.vocab_size = vocab_size
        if no_more_points_class:
            self.no_more_points_token_id = vocab_size + n_patches
        else:
            self.no_more_points_token_id = -1
        self.patch_start = vocab_size
        self.patch_end_without_no_more_points = vocab_size + n_patches
        self.patch_end = vocab_size + n_patches + int(no_more_points_class)
        self.subpatch_start = self.patch_end
        self.subpatch_end = self.subpatch_start + n_subpatches
        self.location_start = self.subpatch_end
        self.location_end = self.subpatch_end + n_locations


class MolmoPointLogitProcessor:
    """Enforce valid point token generation order in MLX."""

    def __init__(
        self,
        bounds: GeneratedTokenBounds,
        prevent_repeats,
        force_patch_sorted,
        force_subpatch_sorted,
    ):
        self.bounds = bounds
        self.prevent_repeats = prevent_repeats
        self.force_patch_sorted = force_patch_sorted
        self.force_subpatch_sorted = force_subpatch_sorted

    def __call__(self, generated_ids_np, last_token_int, vocab_size):
        """Build a logit mask. Uses numpy (no mx.eval sync).

        Args:
            generated_ids_np: list of int token ids generated so far
            last_token_int: int, the last generated token
            vocab_size: int, extended vocabulary size

        Returns:
            mx.array of shape (vocab_size,) with -1e9 at invalid positions
        """
        b = self.bounds
        NEG_INF = np.float32(-1e9)
        mask = np.zeros(vocab_size, dtype=np.float32)

        last_token = last_token_int
        ids = generated_ids_np

        skip = 2 if b.n_locations else 1
        last_patch = None
        last_subpatch = None
        # Check ALL tokens for no_more_points (not subject to skip)
        no_more_points = any(tok == b.no_more_points_token_id for tok in ids)
        # Only scan history up to skip for patch/subpatch tracking
        for i in range(len(ids) - skip):
            tok = ids[i]
            if b.patch_start <= tok < b.patch_end:
                last_patch = tok
            elif b.subpatch_start <= tok < b.subpatch_end:
                last_subpatch = tok

        if no_more_points:
            mask[b.patch_start : b.location_end] = NEG_INF
        elif last_token < b.patch_start or last_token >= b.subpatch_end:
            # Can generate text or a patch, but not subpatch/location
            mask[b.subpatch_start : b.location_end] = NEG_INF
            if self.force_patch_sorted and last_patch is not None:
                # Patches must be in sorted order
                mask[b.patch_start : last_patch] = NEG_INF
            if (
                self.prevent_repeats
                and self.force_subpatch_sorted
                and last_subpatch is not None
                and last_subpatch == (b.subpatch_end - 1)
            ):
                # Last subpatch was at max — selecting same patch would force
                # a repeat since sorted order has no room for a new subpatch
                if last_patch is not None:
                    mask[last_patch] = NEG_INF
        elif b.patch_start <= last_token < b.patch_end:
            # After a patch, must select a subpatch
            mask[: b.subpatch_start] = NEG_INF
            mask[b.subpatch_end :] = NEG_INF
            if (
                self.force_subpatch_sorted
                and last_patch == last_token
                and last_subpatch is not None
            ):
                if self.prevent_repeats:
                    mask[b.subpatch_start : last_subpatch + 1] = NEG_INF
                else:
                    mask[b.subpatch_start : last_subpatch] = NEG_INF
        elif b.n_locations and b.subpatch_start <= last_token < b.subpatch_end:
            # After a subpatch, must select a location
            mask[: b.location_start] = NEG_INF
            mask[b.location_end :] = NEG_INF

        return mx.array(mask)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        vit_config = config.vision_config
        adapter_config = config.adapter_config

        vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                vit_layers.append(layer)
            else:
                vit_layers.append(layer + vit_config.num_hidden_layers)
        self._vit_layers = vit_layers

        last_layer_needed = max(vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            truncated_config = deepcopy(vit_config)
            truncated_config.num_hidden_layers = last_layer_needed
            self.vision_model = VisionModel(truncated_config)
        else:
            self.vision_model = VisionModel(vit_config)

        # Connector
        self.connector = MolmoPointConnector(adapter_config, vit_config)

        # ViT feature embedding for subpatch tokens
        llm_dim = config.text_config.hidden_size
        vit_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.build_vit_embedding = nn.Linear(vit_dim, llm_dim, bias=True)

        # Point predictor
        self.point_predictor = PointPredictor(config)

        # Language model (stored as `lm` so the `language_model` property can
        # return `self` and route generate_step through Model.__call__ for
        # point prediction support)
        self.lm = LanguageModel(config.text_config, config)

        # Image cache (set during prefill, used during generation)
        self._image_cache = None
        self._token_bounds = None
        self._generated_ids_list = []  # plain Python list of ints for logit processor
        self._last_predicted_patch_id = None

    def _build_token_bounds(self, token_pooling):
        n_patches, n_subpatches = token_pooling.shape[-2:]
        total_vocab = (
            self.config.text_config.vocab_size
            + self.config.text_config.additional_vocab_size
        )
        return GeneratedTokenBounds(
            vocab_size=total_vocab,
            n_patches=n_patches,
            n_subpatches=n_subpatches,
            n_locations=9 if self.config.patch_location else 0,
            no_more_points_class=self.config.no_more_points_class,
        )

    def _build_logit_processor(self):
        return MolmoPointLogitProcessor(
            bounds=self._token_bounds,
            prevent_repeats=self.config.mask_repeats in ["all", "inference"],
            force_patch_sorted=self.config.mask_patches in ["always", "inference"],
            force_subpatch_sorted=self.config.mask_subpatches
            in ["always", "inference"],
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ):
        image_token_pooling = kwargs.get("image_token_pooling", None)
        image_grids = kwargs.get("image_grids", None)
        image_num_crops = kwargs.get("image_num_crops", None)
        token_type_ids = kwargs.get("token_type_ids", None)

        if pixel_values is None:
            inputs_embeds = self.lm.model.wte(input_ids)
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        # Reset image cache for new inference
        self._image_cache = None
        self._token_bounds = None
        self._generated_ids_list = []
        self._last_predicted_patch_id = None

        # Build batched images
        images, token_pooling = self._build_batched_images(
            input_ids, pixel_values, image_token_pooling, image_grids, image_num_crops
        )

        # Mark IDs that should not be -1
        safe_ids = mx.where(input_ids != -1, input_ids, 0)
        x = self.lm.model.wte(safe_ids)
        batch_size = x.shape[0]
        dim = x.shape[-1]

        # Process images through ViT
        is_indexable_image_token = input_ids == self.config.image_patch_id
        is_non_indexable_image_token = (
            input_ids == self.config.image_non_indexable_patch_id
        )
        is_image_token = is_indexable_image_token | is_non_indexable_image_token

        B, T, N, D = images.shape
        images_flat = images.reshape(B * T, N, D)
        # Cast to vision model dtype (critical for quantized models)
        vit_dtype = self.vision_model.patch_embedding.weight.dtype
        images_flat = images_flat.astype(vit_dtype)
        vit_image_features = self.vision_model(images_flat)

        features = []
        for layer_idx in self._vit_layers:
            features.append(vit_image_features[layer_idx])
        vit_features = mx.concatenate(features, axis=-1)
        vit_feature_dim = vit_features.shape[-1]

        # Gather features for pooling
        vit_features = vit_features.reshape(batch_size, -1, vit_feature_dim)
        # Index with token_pooling
        clamped_pooling = mx.clip(token_pooling, 0, vit_features.shape[1] - 1)
        # Gather: vit_features[batch, clamped_pooling]
        batch_idx = mx.arange(batch_size)[:, None, None]
        vit_features_gathered = vit_features[batch_idx, clamped_pooling]
        vit_features_gathered = (
            vit_features_gathered
            * (token_pooling >= 0).astype(vit_features_gathered.dtype)[:, :, :, None]
        )
        vit_features_mask = token_pooling >= 0

        # Build sparse features for connector
        # Use numpy ONLY for bool mask -> int index conversion (tiny, no data copy)
        image_features_mask = mx.any(
            vit_features_mask, axis=-1
        )  # (B, n_pooled_patches)

        flat_mask_np = np.array(image_features_mask.reshape(-1))
        valid_indices_np = np.where(flat_mask_np)[0]
        valid_indices = mx.array(valid_indices_np.astype(np.int32))

        vit_features_flat = vit_features_gathered.reshape(
            -1, token_pooling.shape[-1], vit_feature_dim
        )
        vit_features_sparse = vit_features_flat[valid_indices]
        vit_mask_sparse = vit_features_mask.reshape(-1, token_pooling.shape[-1])[
            valid_indices
        ]

        # Apply connector
        image_features = self.connector(vit_features_sparse, vit_mask_sparse)

        # Add image features to embeddings at image token positions (all mx ops)
        flat_is_image_np = np.array(is_image_token.reshape(-1))
        image_indices_np = np.where(flat_is_image_np)[0]
        image_indices = mx.array(image_indices_np.astype(np.int32))

        # Add image features in float32 to avoid float16 overflow
        # (connector output can exceed float16 max 65504).
        # Keep float32 — the LM layers will handle the dtype internally.
        x_flat = x.reshape(-1, dim).astype(mx.float32)
        x_flat = x_flat.at[image_indices].add(
            image_features.reshape(-1, dim).astype(mx.float32)
        )
        x = x_flat.reshape(x.shape)

        # Build subpatch keys from ViT features (not from transformer output)
        pp = self.point_predictor
        subpatch_k = pp.subpatch_k(vit_features_gathered)

        # Count image tokens per batch for offset computation
        n_image_per_batch = is_image_token.sum(axis=-1).astype(mx.int32)
        offsets = mx.concatenate(
            [mx.array([0]), mx.cumsum(n_image_per_batch[:-1], axis=0)]
        )

        # Compute indexable/non-indexable masks for later patch key building
        is_indexable_flat = is_indexable_image_token.reshape(-1).astype(mx.int32)

        # Store partial image cache — patch keys will be built in _prefill_forward
        # after the transformer runs (they need the hidden state, not embeddings)
        self._image_cache = {
            "subpatch_k": subpatch_k,
            "token_pooling": token_pooling,
            "vit_features": vit_features_gathered,
            "vit_features_mask": vit_features_mask,
            "image_features_mask": image_features_mask,
            "image_features": image_features,
            "image_token_offsets": offsets,
            # Indices needed for building patch keys after transformer
            "image_indices": image_indices,
            "valid_indices": valid_indices,
            "is_indexable_flat": is_indexable_flat,
            "is_image_token": is_image_token,
            "is_indexable_image_token": is_indexable_image_token,
        }

        self._token_bounds = self._build_token_bounds(token_pooling)

        return InputEmbeddingsFeatures(inputs_embeds=x)

    def _build_batched_images(
        self, input_ids, pixel_values, image_token_pooling, image_grids, image_num_crops
    ):
        """Build batched images and token pooling from inputs."""
        batch_size = input_ids.shape[0]

        # Ensure int32 for scatter operations
        image_token_pooling = image_token_pooling.astype(mx.int32)
        image_grids = image_grids.astype(mx.int32)
        image_num_crops = image_num_crops.astype(mx.int32)

        # Count images per example
        raw_counts = (input_ids == self.config.image_end_token_id).sum(axis=1)
        counts = raw_counts // 2  # global + high-res per image

        n_crops, n_patches, pixels_per_patch = pixel_values.shape

        # Compute per-image pooled patch count
        first_prod = image_grids[:, :2].prod(axis=1)
        second_prod = image_grids[:, 2:].prod(axis=1)
        num_pooled_per_image = (first_prod + second_prod).astype(mx.int32)

        # Per-image crop counts and offsets
        counts_list = counts.tolist()
        crops_per_example = []
        offset = 0
        index_offsets_per_example = []
        for c in counts_list:
            c = int(c)
            per_img_crops = image_num_crops[offset : offset + c]
            crops_per_example.append(int(per_img_crops.sum().item()))
            patches_per_img = per_img_crops * n_patches
            idx_offsets = [0]
            for j in range(c - 1):
                idx_offsets.append(idx_offsets[-1] + int(patches_per_img[j].item()))
            index_offsets_per_example.append(idx_offsets)
            offset += c

        num_pooled_per_example = []
        img_offset = 0
        for c in counts_list:
            c = int(c)
            num_pooled_per_example.append(
                int(num_pooled_per_image[img_offset : img_offset + c].sum().item())
            )
            img_offset += c

        M = max(crops_per_example)
        images = mx.full(
            (batch_size, M, n_patches, pixels_per_patch), -1, dtype=pixel_values.dtype
        )
        offset_crop = 0
        for i in range(batch_size):
            num = crops_per_example[i]
            images = images.at[i, :num].add(
                pixel_values[offset_crop : offset_crop + num] - images[i, :num]
            )
            offset_crop += num

        P = max(num_pooled_per_example)
        pool_dim = image_token_pooling.shape[-1]
        new_token_pooling = mx.full((batch_size, P, pool_dim), -1, dtype=mx.int32)
        patch_offset = 0
        img_off = 0
        for i, c in enumerate(counts_list):
            c = int(c)
            n_pooled = num_pooled_per_example[i]
            cur = image_token_pooling[patch_offset : patch_offset + n_pooled]

            # Apply per-image offsets
            per_img_pooled = num_pooled_per_image[img_off : img_off + c]
            idx_offsets = index_offsets_per_example[i]
            sub_offset = 0
            for j in range(c):
                idx_off = idx_offsets[j]
                n = int(per_img_pooled[j].item())
                cur_slice = cur[sub_offset : sub_offset + n]
                cur = cur.at[sub_offset : sub_offset + n].add(
                    mx.where(cur_slice >= 0, idx_off, 0)
                )
                sub_offset += n

            new_token_pooling = new_token_pooling.at[i, :n_pooled].add(
                cur - new_token_pooling[i, :n_pooled]
            )
            patch_offset += n_pooled
            img_off += c

        return images, new_token_pooling

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        """Main forward pass. Used by generate_step.

        During prefill: pixel_values are processed, inputs_embeds contain image features.
        During generation: input_ids are token IDs (possibly in extended vocab range).
        """
        is_generating = (
            (self._image_cache is not None)
            and (inputs_embeds is None)
            and (input_ids is not None)
        )

        if is_generating:
            # Autoregressive generation with image cache
            return self._generate_forward(input_ids, mask, cache)
        else:
            # Prefill or text-only
            return self._prefill_forward(input_ids, inputs_embeds, mask, cache)

    def _prefill_forward(self, input_ids, inputs_embeds, mask, cache):
        """Standard forward pass for prefill.

        After the transformer runs, build point predictor patch keys from
        the pre-LN hidden state (matching the original torch implementation).
        """
        h, pre_ln_h = self.lm.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            return_pre_ln=True,
        )
        logits = self.lm.lm_head(h)

        # Build point predictor cache from transformer hidden states
        if self._image_cache is not None and "patch_k" not in self._image_cache:
            ic = self._image_cache
            pp = self.point_predictor
            dim = self.config.text_config.hidden_size
            batch_size = pre_ln_h.shape[0]
            n_pooled = ic["token_pooling"].shape[1]

            image_indices = ic["image_indices"]
            valid_indices = ic["valid_indices"]
            is_indexable_flat = ic["is_indexable_flat"]

            # Compute patch keys from the HIDDEN STATE (not embeddings)
            x_norm = (
                pp.x_norm(pre_ln_h)
                if pp.x_norm is not None
                else pre_ln_h / math.sqrt(dim)
            )
            x_norm_flat = x_norm.reshape(-1, dim)
            patch_k_flat = pp.patch_k(x_norm_flat[image_indices])

            if pp.patch_rotary is not None:
                image_token_cumsum = mx.cumsum(is_indexable_flat, axis=0) - 1
                image_pos_ids_flat = image_token_cumsum[image_indices]
                patch_k_flat = pp.patch_rotary(patch_k_flat, image_pos_ids_flat)

                image_pos_ids = mx.zeros((batch_size * n_pooled,), dtype=mx.int32)
                image_pos_ids = image_pos_ids.at[valid_indices].add(image_pos_ids_flat)
                image_pos_ids = image_pos_ids.reshape(batch_size, n_pooled)
            else:
                image_pos_ids = None

            # Build patch_k tensor via scatter
            patch_k = mx.zeros(
                (batch_size * n_pooled, patch_k_flat.shape[-1]), dtype=pre_ln_h.dtype
            )
            patch_k = patch_k.at[valid_indices].add(patch_k_flat.astype(pre_ln_h.dtype))
            patch_k = patch_k.reshape(batch_size, n_pooled, -1)

            # Build patch_k_mask
            is_indexable_at_image = is_indexable_flat[image_indices]
            patch_k_mask = mx.zeros((batch_size * n_pooled,), dtype=mx.int32)
            patch_k_mask = patch_k_mask.at[valid_indices].add(is_indexable_at_image)
            patch_k_mask = patch_k_mask.reshape(batch_size, n_pooled).astype(mx.bool_)

            if self.config.no_more_points_class:
                patch_k = pp.add_no_point_class_embed(patch_k)
                patch_k_mask = mx.concatenate(
                    [patch_k_mask, mx.ones((batch_size, 1), dtype=mx.bool_)], axis=1
                )

            ic["patch_k"] = patch_k
            ic["patch_k_mask"] = patch_k_mask
            ic["image_pos_ids"] = image_pos_ids

            # Extend logits with dummy point logits for the prefill step
            B, S, V = logits.shape
            bounds = self._token_bounds
            total_extra = bounds.location_end - bounds.patch_start
            dummy_logits = mx.full((B, S, total_extra), -100000.0, dtype=logits.dtype)
            logits = mx.concatenate([logits, dummy_logits], axis=-1)

        return LanguageModelOutput(logits=logits)

    def _generate_forward(self, input_ids, mask, cache):
        """Forward pass during autoregressive generation with point prediction."""
        bounds = self._token_bounds
        ic = self._image_cache
        pp = self.point_predictor
        dim = self.config.text_config.hidden_size
        batch_size = input_ids.shape[0]

        # Track generated IDs as plain Python ints (no mx.eval sync)
        for i in range(input_ids.shape[1]):
            self._generated_ids_list.append(int(input_ids[0, i].item()))

        # Cast input_ids to int32 to avoid uint32 overflow issues
        input_ids_i32 = input_ids.astype(mx.int32)

        # Decode extended tokens back to original special token IDs
        is_patch = (input_ids_i32 >= bounds.patch_start) & (
            input_ids_i32 < bounds.patch_end_without_no_more_points
        )
        is_no_more_points = input_ids_i32 == bounds.no_more_points_token_id
        is_subpatch = (input_ids_i32 >= bounds.subpatch_start) & (
            input_ids_i32 < bounds.subpatch_end
        )
        is_location = (input_ids_i32 >= bounds.location_start) & (
            input_ids_i32 < bounds.location_end
        )

        input_patch_ids = mx.where(is_patch, input_ids_i32 - bounds.patch_start, -1)
        input_subpatch_ids = mx.where(
            is_subpatch, input_ids_i32 - bounds.subpatch_start, -1
        )

        # Map extended tokens back to original special token IDs for embedding
        decoded_ids = input_ids_i32
        decoded_ids = mx.where(
            is_patch | is_no_more_points, self.config.patch_token_id, decoded_ids
        )
        decoded_ids = mx.where(is_subpatch, self.config.subpatch_token_id, decoded_ids)
        decoded_ids = mx.where(is_location, self.config.location_token_id, decoded_ids)

        # Embed the tokens
        x = self.lm.model.wte(decoded_ids)

        # Add image features for patch tokens
        any_patch = mx.any(is_patch).item()
        if any_patch:
            img_features = ic["image_features"]
            offsets = ic["image_token_offsets"]
            for b in range(batch_size):
                pid = int(input_patch_ids[b, 0].item())
                if (
                    pid >= 0
                    and pid
                    < bounds.patch_end_without_no_more_points - bounds.patch_start
                ):
                    flat_idx = pid + int(offsets[b].item())
                    feat = img_features.reshape(-1, dim)[flat_idx : flat_idx + 1]
                    x = x.at[b, 0].add(feat[0])

        # Embed subpatch tokens with ViT features (all mx ops, no numpy)
        any_subpatch = mx.any(is_subpatch).item()
        if any_subpatch:
            vit_features = ic["vit_features"]
            offsets = ic["image_token_offsets"]
            feat_mask_np = np.array(ic["image_features_mask"].reshape(-1))
            valid_indices_np = np.where(feat_mask_np)[0]
            valid_indices = mx.array(valid_indices_np.astype(np.int32))
            vit_feat = vit_features.reshape(
                -1, ic["token_pooling"].shape[-1], vit_features.shape[-1]
            )
            vit_sparse = vit_feat[valid_indices]
            for b in range(batch_size):
                spid = int(input_subpatch_ids[b, 0].item())
                if spid >= 0 and self._last_predicted_patch_id is not None:
                    lpid = int(self._last_predicted_patch_id[b].item())
                    flat_pid = lpid + int(offsets[b].item())
                    vit_to_embed = vit_sparse[flat_pid, spid : spid + 1]
                    embedded = self.build_vit_embedding(vit_to_embed)
                    # Replace embedding in-place using mx scatter
                    zeros = mx.zeros_like(x[b, 0:1])
                    x = x.at[b, 0:1].add(embedded - x[b, 0:1])

        # Run through transformer
        h, pre_ln_h = self.lm.model(
            inputs_embeds=x,
            mask=mask,
            cache=cache,
            return_pre_ln=True,
        )

        # Compute standard logits
        logits = self.lm.lm_head(h)

        # Point predictor
        x_norm = (
            pp.x_norm(pre_ln_h) if pp.x_norm is not None else pre_ln_h / math.sqrt(dim)
        )

        # Patch logits
        image_q = pp.patch_q(x_norm)
        if pp.patch_rotary is not None and self._last_predicted_patch_id is not None:
            pos_ids = ic["image_pos_ids"]
            batch_idx = mx.arange(batch_size)
            lpid = self._last_predicted_patch_id
            rotate_by = pos_ids[
                batch_idx, mx.clip(lpid.squeeze(-1), 0, pos_ids.shape[1] - 1)
            ]
            rotate_by = mx.where(lpid.squeeze(-1) >= 0, rotate_by, 0)
            image_q_flat = image_q.reshape(-1, image_q.shape[-1])
            image_q_flat = pp.patch_rotary(
                image_q_flat, mx.clip(rotate_by, a_min=0, a_max=None)
            )
            image_q = image_q_flat.reshape(batch_size, -1, image_q.shape[-1])

        dots = image_q @ ic["patch_k"].transpose(0, 2, 1)
        if self.config.norm_logits:
            dots = dots / math.sqrt(dots.shape[-1])
        valid = ic["patch_k_mask"][:, None, :]
        patch_logits = mx.where(valid, dots, -100000.0)

        # Replace patch_token_id logit in main logits with argmax patch score
        B, S, V = logits.shape
        patch_token_logits = logits[
            :, :, self.config.patch_token_id : self.config.patch_token_id + 1
        ]
        logits = logits.at[:, :, self.config.patch_token_id].add(
            -100000.0 - logits[:, :, self.config.patch_token_id]
        )

        n_patches = patch_logits.shape[-1]
        # Vectorized: place patch_token_logits at the argmax patch position
        selected_patches = mx.argmax(patch_logits, axis=-1)  # (B, S)
        argmax_patch_logits = mx.full((B, S, n_patches), -100000.0, dtype=logits.dtype)
        # Manual one_hot: compare indices to selected_patches
        indices = mx.arange(n_patches)[None, None, :]  # (1, 1, n_patches)
        is_selected = indices == selected_patches[:, :, None]  # (B, S, n_patches)
        argmax_patch_logits = mx.where(
            is_selected, patch_token_logits, argmax_patch_logits
        )

        # Subpatch logits
        n_subpatches = ic["token_pooling"].shape[-1]
        subpatch_logits = mx.full((B, S, n_subpatches), -100000.0, dtype=logits.dtype)
        if any_patch:
            subpatch_point_q = pp.subpatch_q(
                x_norm.squeeze(1) if S == 1 else x_norm[:, -1:].squeeze(1)
            )
            batch_idx = mx.arange(batch_size)
            spk = ic["subpatch_k"][
                batch_idx,
                mx.clip(input_patch_ids.squeeze(1), 0, ic["subpatch_k"].shape[1] - 1),
            ]
            sp_logits = mx.sum(subpatch_point_q[:, None, :] * spk, axis=-1)
            if self.config.norm_logits:
                sp_logits = sp_logits / math.sqrt(ic["patch_k"].shape[-1])
            sp_mask = ic["vit_features_mask"][
                batch_idx,
                mx.clip(
                    input_patch_ids.squeeze(1), 0, ic["vit_features_mask"].shape[1] - 1
                ),
            ]
            sp_logits = mx.where(sp_mask, sp_logits, -100000.0)
            subpatch_logits = sp_logits[:, None, :]

        # Suppress subpatch_token_id in main logits
        logits = logits.at[:, :, self.config.subpatch_token_id].add(
            -100000.0 - logits[:, :, self.config.subpatch_token_id]
        )

        # Location logits
        location_logits = mx.full((B, S, 9), -100000.0, dtype=logits.dtype)
        any_subpatch_item = mx.any(is_subpatch).item()
        if any_subpatch_item:
            location_logits = pp.subpatch_loc_k(pre_ln_h)

        # Suppress location_token_id in main logits
        logits = logits.at[:, :, self.config.location_token_id].add(
            -100000.0 - logits[:, :, self.config.location_token_id]
        )

        # Concatenate extended logits
        logits = mx.concatenate(
            [logits, argmax_patch_logits, subpatch_logits, location_logits], axis=-1
        )

        # Apply logit processor (uses numpy mask, no mx.eval sync)
        if self._generated_ids_list:
            processor = self._build_logit_processor()
            last_tok = self._generated_ids_list[-1]
            lp_mask = processor(self._generated_ids_list, last_tok, logits.shape[-1])
            # Add mask to last position logits
            last_logits = logits[:, -1, :] + lp_mask
            logits = mx.concatenate(
                [logits[:, :-1, :], last_logits[:, None, :]], axis=1
            )

        # Update last_predicted_patch_id
        if mx.any(input_patch_ids >= 0).item():
            self._last_predicted_patch_id = mx.where(
                input_patch_ids == -1,
                (
                    self._last_predicted_patch_id
                    if self._last_predicted_patch_id is not None
                    else mx.array([[-1]] * batch_size)
                ),
                input_patch_ids,
            )

        return LanguageModelOutput(logits=logits)

    @property
    def language_model(self):
        """Return self so generate_step routes through Model.__call__
        which includes point prediction logic."""
        return self

    @property
    def layers(self):
        return self.lm.layers

    @property
    def head_dim(self):
        return self.config.text_config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.text_config.num_key_value_heads

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            new_k = k

            # HF checkpoint keys start with "model."
            if new_k.startswith("model."):
                new_k = new_k[len("model.") :]

            # lm_head -> lm.lm_head
            if new_k.startswith("lm_head."):
                new_k = "lm." + new_k

            # LLM transformer -> lm.model
            if new_k.startswith("transformer."):
                new_k = "lm.model." + new_k[len("transformer.") :]

            # ViT: vit.transformer.resblocks -> vision_model.resblocks
            new_k = new_k.replace("vit.transformer.resblocks", "vision_model.resblocks")
            # ViT: vit.* -> vision_model.* (remaining keys)
            if new_k.startswith("vit."):
                new_k = "vision_model." + new_k[len("vit.") :]

            # Cast float32 weights to float16 (HF checkpoint is float32)
            if v.dtype == mx.float32:
                v = v.astype(mx.float16)

            sanitized[new_k] = v
        return sanitized
