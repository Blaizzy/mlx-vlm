import math
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from ..base import InputEmbeddingsFeatures
from .anyup import AnyUp
from .config import ModelConfig, VisionConfig
from .language import LanguageModel, compute_pos_hw, create_falcon_perception_mask


class FourierEncoder(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, feat_dim // 2, bias=False)
        self.transform = nn.Linear(feat_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        f = 2 * math.pi * self.embed(x)
        f = mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)
        return self.transform(f)


class BboxDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.relu(self.w1(x)) ** 2)


class SegmDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.layers = [nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)]
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.relu(layer(x)) ** 2
        return self.pixel_layer(x)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig = None):
        super().__init__()


class Model(nn.Module):
    no_chunked_prefill = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config, config)

        hidden_size = config.text_config.hidden_size

        self.coord_encoder = FourierEncoder(2, config.coord_enc_dim, hidden_size)
        self.coord_decoder = BboxDecoder(
            hidden_size, config.coord_dec_dim, config.coord_out_dim
        )
        self.size_encoder = FourierEncoder(2, config.size_enc_dim, hidden_size)
        self.size_decoder = BboxDecoder(
            hidden_size, config.size_dec_dim, config.size_out_dim
        )

        if config.do_segmentation:
            self.proj_segm = SegmDecoder(
                hidden_size, config.segm_out_dim, config.num_segm_layers
            )
            self.conv_segm = nn.Conv2d(
                hidden_size, config.segm_out_dim, kernel_size=3, padding=1
            )
            self.itok_upsampler = AnyUp(input_dim=3, qk_dim=128, num_heads=4)

        # Give LM direct refs to perception heads (not circular — leaf modules)
        lm = self.language_model
        object.__setattr__(lm, "_coord_encoder", self.coord_encoder)
        object.__setattr__(lm, "_coord_decoder", self.coord_decoder)
        object.__setattr__(lm, "_size_encoder", self.size_encoder)
        object.__setattr__(lm, "_size_decoder", self.size_decoder)
        object.__setattr__(lm, "_perception_config", config)
        if config.do_segmentation:
            object.__setattr__(lm, "_proj_segm", self.proj_segm)
        # Weak-ish ref for segm features (stored on Model, not a module)
        object.__setattr__(lm, "_parent_model_ref", self)

        # Initialize segm state so LM forward works before get_input_embeddings
        self._segm_features_computed = False
        self._segm_features = None
        self._orig_hw = None
        self._prefill_hidden_state = None
        self._prefill_pixel_values = None
        self._prefill_grid_hw = None
        self._prefill_input_ids = None

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_hw = kwargs.get("image_grid_hw", None)

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # New image — reset detection state
        self.language_model._detections = []
        self.language_model._current_det = {}
        self.language_model._pending_coord_xy = None
        self.language_model._pending_size_hw = None

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
        else:
            hidden_states = self._patchify_and_project(pixel_values)

        final_embeds = self._merge_image_features(
            self.config.img_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        position_ids, pos_hw, delta = self._precompute_positions(
            input_ids, image_grid_hw
        )
        self.language_model._position_ids = position_ids
        self.language_model._pos_hw = pos_hw
        self.language_model._rope_delta = delta
        single_ids = input_ids[0:1] if input_ids.ndim == 2 else input_ids
        self.language_model._full_attn_mask = create_falcon_perception_mask(
            single_ids,
            self.config.image_cls_token_id,
            self.config.img_end_id,
        )

        # Store info for lazy segm feature computation (after first LM forward)
        self._prefill_input_ids = input_ids
        self._prefill_pixel_values = pixel_values
        self._prefill_grid_hw = (
            (int(image_grid_hw[0, 0].item()), int(image_grid_hw[0, 1].item()))
            if image_grid_hw is not None
            else None
        )
        self._prefill_hidden_state = None  # set after first forward
        self._segm_features_computed = False
        self._segm_features = None
        self._orig_hw = None

        return InputEmbeddingsFeatures(inputs_embeds=final_embeds)

    def _precompute_positions(self, input_ids, image_grid_hw):
        single_ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        ids = single_ids.reshape(-1).tolist()
        start_id = self.config.image_cls_token_id
        end_id = self.config.img_end_id

        pos_t = []
        in_image = False
        next_pos = 0
        for tok in ids:
            if tok == start_id and not in_image:
                in_image = True
            pos_t.append(next_pos)
            if not in_image:
                next_pos += 1
            if tok == end_id and in_image:
                in_image = False
                next_pos += 1

        position_ids = mx.array(pos_t, dtype=mx.int32)
        delta = int(mx.max(position_ids).item()) + 1 - len(ids)

        grid_hws = None
        if image_grid_hw is not None:
            if isinstance(image_grid_hw, mx.array):
                grid_hws = image_grid_hw.tolist()
            elif isinstance(image_grid_hw, list):
                grid_hws = image_grid_hw
            if grid_hws:
                grid_hws = [tuple(int(x) for x in g) for g in grid_hws]
                if input_ids.ndim == 2:
                    grid_hws = grid_hws[:1]

        pos_hw = compute_pos_hw(
            single_ids.reshape(-1),
            image_token_id=self.config.img_id,
            image_grid_hws=grid_hws,
        )

        return position_ids, pos_hw, delta

    def _patchify_and_project(self, pixel_values: mx.array) -> mx.array:
        ps = self.config.vision_config.spatial_patch_size
        pt = self.config.vision_config.temporal_patch_size

        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]

        N, H, W, C = pixel_values.shape
        h_patches = H // ps
        w_patches = W // ps

        patches = pixel_values.reshape(N, h_patches, ps, w_patches, ps, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(N * h_patches * w_patches, ps * ps * C * pt)

        return self.language_model.model.img_projector(
            patches.astype(self.language_model.model.img_projector.weight.dtype)
        )

    @staticmethod
    def _merge_image_features(
        image_token_id: int,
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        image_positions = input_ids == image_token_id

        batch_outputs = []
        feature_start = 0

        for b in range(batch_size):
            mask = image_positions[b]
            n_pos = mx.sum(mask).item()

            if n_pos > 0:
                batch_feats = image_features[feature_start : feature_start + n_pos]
                if batch_feats.shape[0] != n_pos:
                    raise ValueError(
                        f"Image token positions ({n_pos}) does not match "
                        f"image features ({batch_feats.shape[0]}) for batch {b}"
                    )
                cumsum = mx.cumsum(mask.astype(mx.int32))
                feat_idx = mx.where(mask, cumsum - 1, 0)
                gathered = batch_feats[feat_idx]
                mask_exp = mx.expand_dims(mask, axis=-1)
                batch_out = mx.where(mask_exp, gathered, inputs_embeds[b])
                feature_start += n_pos
            else:
                batch_out = inputs_embeds[b]

            batch_outputs.append(batch_out)

        return mx.stack(batch_outputs, axis=0)

    def encode_coords_into_embeds(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        coord_xy: Optional[mx.array] = None,
    ) -> mx.array:
        if coord_xy is None:
            return inputs_embeds
        coord_mask = input_ids == self.config.coord_token_id
        if not mx.any(coord_mask).item():
            return inputs_embeds
        coord_tokens = self.coord_encoder(coord_xy.reshape(-1, 2))
        coord_tokens = coord_tokens.reshape(
            inputs_embeds.shape[0], -1, inputs_embeds.shape[-1]
        )
        mask_exp = mx.expand_dims(coord_mask, axis=-1)
        return mx.where(mask_exp, coord_tokens, inputs_embeds)

    def encode_sizes_into_embeds(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        size_hw: Optional[mx.array] = None,
    ) -> mx.array:
        if size_hw is None:
            return inputs_embeds
        size_mask = input_ids == self.config.size_token_id
        if not mx.any(size_mask).item():
            return inputs_embeds
        size_tokens = self.size_encoder(size_hw.reshape(-1, 2))
        size_tokens = size_tokens.reshape(
            inputs_embeds.shape[0], -1, inputs_embeds.shape[-1]
        )
        mask_exp = mx.expand_dims(size_mask, axis=-1)
        return mx.where(mask_exp, size_tokens, inputs_embeds)

    def decode_coords(self, hidden_state: mx.array) -> mx.array:
        logits = self.coord_decoder(hidden_state)
        half = self.config.coord_out_dim // 2
        return logits.reshape(-1, 2, half)

    def decode_sizes(self, hidden_state: mx.array) -> mx.array:
        logits = self.size_decoder(hidden_state)
        half = self.config.size_out_dim // 2
        return logits.reshape(-1, 2, half)

    @staticmethod
    def process_sizes(logits: mx.array) -> mx.array:
        num_bins = logits.shape[-1]
        pred = mx.argmax(logits, axis=-1).astype(mx.float32) / (num_bins - 1)
        min_size = math.log2(1.0 / num_bins)
        max_size = 0.0
        pred = pred * (max_size - min_size) + min_size
        return 2.0**pred

    def compute_segm_features(
        self,
        hidden_state: mx.array,
        input_ids: mx.array,
        pixel_values: mx.array,
        grid_h: int,
        grid_w: int,
    ) -> mx.array:
        """Extract image token hidden states and produce high-res segm features via AnyUp.

        Returns: (1, H, W, segm_out_dim) high-res features at original image resolution
        """
        img_mask = input_ids[0] == self.config.img_id
        n_img = mx.sum(img_mask).item()
        expected = grid_h * grid_w
        if n_img != expected:
            raise ValueError(
                f"Image tokens ({n_img}) != grid ({grid_h}x{grid_w}={expected})"
            )

        # Gather image hidden states
        indices = mx.array([i for i, v in enumerate(img_mask.tolist()) if v])
        img_features = hidden_state[0, indices]  # (n_img, D)
        img_features = img_features.reshape(1, grid_h, grid_w, -1)  # (1, h, w, D)

        # Low-res segm features
        lr_features = self.conv_segm(img_features)  # (1, h, w, segm_out_dim)

        # Upsample to high-res via AnyUp
        if hasattr(self, "itok_upsampler"):
            images = pixel_values
            if images.ndim == 3:
                images = images[None]

            _, H, W, _ = images.shape
            ps = self.config.vision_config.spatial_patch_size

            # Match PyTorch pipeline: pad images and features to multiples
            # of ps so AnyUp sees the same spatial context as training.
            max_dim = max(H, W)
            pad_h = ((max_dim + ps - 1) // ps) * ps
            pad_w = pad_h
            if pad_h != H or pad_w != W:
                images = mx.pad(
                    images, [(0, 0), (0, pad_h - H), (0, pad_w - W), (0, 0)]
                )
                gh_pad = pad_h // ps
                gw_pad = pad_w // ps
                lr_features = mx.pad(
                    lr_features,
                    [(0, 0), (0, gh_pad - grid_h), (0, gw_pad - grid_w), (0, 0)],
                )

            hr_features = self.itok_upsampler(images, lr_features)

            # Crop back to actual image region
            if pad_h != H or pad_w != W:
                hr_features = hr_features[:, :H, :W, :]

            return hr_features

        return lr_features

    @staticmethod
    def _bilinear_upsample(x: mx.array, out_h: int, out_w: int) -> mx.array:
        """Bilinear upsample a 2D array from (h, w) to (out_h, out_w)."""
        in_h, in_w = x.shape

        # Map output coords to input space (align_corners=False style)
        y = (mx.arange(out_h).astype(mx.float32) + 0.5) * (in_h / out_h) - 0.5
        xc = (mx.arange(out_w).astype(mx.float32) + 0.5) * (in_w / out_w) - 0.5

        y = mx.clip(y, 0, in_h - 1.0)
        xc = mx.clip(xc, 0, in_w - 1.0)

        y0 = mx.floor(y).astype(mx.int32)
        x0 = mx.floor(xc).astype(mx.int32)
        y1 = mx.minimum(y0 + 1, in_h - 1)
        x1 = mx.minimum(x0 + 1, in_w - 1)

        wy = (y - y0.astype(mx.float32))[:, None]  # (out_h, 1)
        wx = (xc - x0.astype(mx.float32))[None, :]  # (1, out_w)

        val_00 = x[y0][:, x0]
        val_01 = x[y0][:, x1]
        val_10 = x[y1][:, x0]
        val_11 = x[y1][:, x1]

        return (
            (1 - wy) * (1 - wx) * val_00
            + (1 - wy) * wx * val_01
            + wy * (1 - wx) * val_10
            + wy * wx * val_11
        )

    def decode_segm_mask(
        self,
        seg_hidden: mx.array,
        segm_features: mx.array,
        orig_h: int,
        orig_w: int,
        threshold: float = 0.5,
    ) -> mx.array:
        """Decode a segmentation mask from a seg token's hidden state.

        Args:
            seg_hidden: (D,) hidden state at the seg token position
            segm_features: (1, feat_h, feat_w, segm_out_dim) from compute_segm_features
            orig_h, orig_w: original image dimensions
            threshold: sigmoid threshold for binary mask

        Returns: (orig_h, orig_w) binary mask
        """
        seg_token = self.proj_segm(seg_hidden)  # (segm_out_dim,)

        # Dot product: features (feat_h, feat_w, D) x token (D,) → (feat_h, feat_w)
        mask_logits = mx.sum(segm_features[0] * seg_token[None, None, :], axis=-1)

        feat_h, feat_w = mask_logits.shape
        if feat_h == orig_h and feat_w == orig_w:
            # High-res features from AnyUp - no upsampling needed
            return mx.sigmoid(mask_logits) > threshold

        # Low-res fallback - bilinear upsample
        upsampled = self._bilinear_upsample(mask_logits, orig_h, orig_w)
        return mx.sigmoid(upsampled) > threshold

    def _ensure_segm_features(self):
        """Lazily compute AnyUp segm features from prefill hidden states."""
        if self._segm_features_computed or not hasattr(self, "conv_segm"):
            return
        if self._prefill_hidden_state is None or self._prefill_grid_hw is None:
            return
        grid_h, grid_w = self._prefill_grid_hw
        self._segm_features = self.compute_segm_features(
            self._prefill_hidden_state,
            self._prefill_input_ids,
            self._prefill_pixel_values,
            grid_h,
            grid_w,
        )
        self._orig_hw = (
            self._prefill_pixel_values.shape[-3],
            self._prefill_pixel_values.shape[-2],
        )
        mx.eval(self._segm_features)
        self._segm_features_computed = True
        # Free prefill data after computing features
        self._prefill_hidden_state = None
        self._prefill_pixel_values = None

    def get_detections(self):
        """Return detections accumulated during the last generate() call.

        Each detection has 'xy' (center coords), 'hw' (size), and optionally 'mask'.
        """
        lm = self.language_model
        # Flush any pending detection
        if "xy" in lm._current_det and "hw" in lm._current_det:
            lm._detections.append(lm._current_det)
            lm._current_det = {}
        return lm._detections

    @property
    def last_hidden_state(self) -> Optional[mx.array]:
        return self.language_model.model._last_hidden_state

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        coord_xy = kwargs.pop("coord_xy", None)
        size_hw = kwargs.pop("size_hw", None)

        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        embeds = features.inputs_embeds

        embeds = self.encode_coords_into_embeds(embeds, input_ids, coord_xy)
        embeds = self.encode_sizes_into_embeds(embeds, input_ids, size_hw)

        kwargs["pixel_values"] = pixel_values
        out = self.language_model(
            input_ids,
            inputs_embeds=embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

        # Save prefill hidden state for lazy segm features
        if pixel_values is not None and not self._segm_features_computed:
            h = self.last_hidden_state
            if h is not None:
                self._prefill_hidden_state = h
                mx.eval(self._prefill_hidden_state)

        return out

    def generate_perception(
        self,
        processor,
        *,
        image,
        query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        segm_threshold: float = 0.5,
    ) -> List[Dict]:
        """Run Falcon Perception detection with proper coord/size/seg decode loop.

        Returns list of detections, each with:
          - 'xy': dict with 'x', 'y' (center coords, normalized 0-1)
          - 'hw': dict with 'h', 'w' (size, as fraction of image)
          - 'mask': (H, W) binary mx.array segmentation mask (if seg token decoded)
        """
        from mlx_vlm.utils import load_image as _load_image

        from ..base import to_mlx
        from ..cache import make_prompt_cache

        if not isinstance(image, Image.Image):
            image = _load_image(image)
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        result = processor(text=[query], images=[image], padding=False)
        result = to_mlx(result)
        input_ids = result["input_ids"]
        pixel_values = result["pixel_values"]
        image_grid_hw = result.get("image_grid_hw", None)

        config = self.config
        coord_token_id = config.coord_token_id
        size_token_id = config.size_token_id
        seg_token_id = config.seg_token_id
        eos_id = config.eos_id

        # Get grid dimensions for segmentation
        if image_grid_hw is not None:
            grid_h = int(image_grid_hw[0, 0].item())
            grid_w = int(image_grid_hw[0, 1].item())
        else:
            ps = config.vision_config.spatial_patch_size
            grid_h = orig_h // ps
            grid_w = orig_w // ps

        cache = make_prompt_cache(self)

        # Prefill: run full model with image
        logits_out = self(
            input_ids,
            pixel_values=pixel_values,
            cache=cache,
            image_grid_hw=image_grid_hw,
        )
        logits = logits_out.logits

        # Compute segmentation features from prefill hidden states (via AnyUp if available)
        segm_features = None
        if hasattr(self, "conv_segm"):
            prefill_hidden = self.last_hidden_state
            if prefill_hidden is not None:
                segm_features = self.compute_segm_features(
                    prefill_hidden, input_ids, pixel_values, grid_h, grid_w
                )
                mx.eval(segm_features)

        # After prefill, clear _position_ids so decode falls back to _rope_delta.
        # _position_ids is sized for prefill only; indexing it at decode offsets fails.
        lm = self.language_model
        lm._position_ids = None
        lm._pos_hw = None
        # _rope_delta and _full_attn_mask are kept for correct decode positions.

        embed_fn = lm.model.embed_tokens

        detections = []
        current_det = {}
        coord_xy = mx.zeros((1, 2))
        size_hw_val = mx.zeros((1, 2))

        for step in range(max_new_tokens):
            if temperature == 0.0:
                token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                logits_scaled = logits[:, -1, :] / temperature
                token = mx.random.categorical(logits_scaled)

            token_id = token.item()

            if token_id == eos_id:
                break

            token_2d = token.reshape(1, 1)

            # Decode coord/size/seg from the hidden state of the PREVIOUS step
            h_state = self.last_hidden_state
            h_last = h_state[:, -1:, :] if h_state is not None else None

            if token_id == coord_token_id and h_last is not None:
                # Commit previous detection before starting a new one
                if "xy" in current_det and "hw" in current_det:
                    detections.append(current_det)
                    current_det = {}

                coord_logits = self.decode_coords(h_last.reshape(-1, h_last.shape[-1]))
                num_bins = coord_logits.shape[-1]
                pred_bins = mx.argmax(coord_logits, axis=-1)
                pred_x = pred_bins[0, 0].item() / (num_bins - 1)
                pred_y = pred_bins[0, 1].item() / (num_bins - 1)
                coord_xy = mx.array([[pred_x, pred_y]])
                current_det["xy"] = {"x": pred_x, "y": pred_y}

            elif token_id == size_token_id and h_last is not None:
                size_logits = self.decode_sizes(h_last.reshape(-1, h_last.shape[-1]))
                hw_pred = self.process_sizes(size_logits)
                pred_h = hw_pred[0, 0].item()
                pred_w = hw_pred[0, 1].item()
                size_hw_val = mx.array([[pred_h, pred_w]])
                current_det["hw"] = {"h": pred_h, "w": pred_w}

            elif token_id == seg_token_id and h_last is not None:
                if segm_features is not None:
                    seg_hidden = h_last.reshape(-1)  # (D,)
                    mask = self.decode_segm_mask(
                        seg_hidden, segm_features, orig_h, orig_w, segm_threshold
                    )
                    mx.eval(mask)
                    current_det["mask"] = mask
                if "xy" in current_det and "hw" in current_det:
                    detections.append(current_det)
                current_det = {}

            # Decode step: embed token, apply coord/size encoding, run LM
            embeds = embed_fn(token_2d)
            embeds = self.encode_coords_into_embeds(
                embeds,
                token_2d,
                coord_xy if token_id == coord_token_id else None,
            )
            embeds = self.encode_sizes_into_embeds(
                embeds,
                token_2d,
                size_hw_val if token_id == size_token_id else None,
            )
            logits_out = lm(
                token_2d,
                inputs_embeds=embeds,
                cache=cache,
            )
            logits = logits_out.logits
            mx.eval(logits)

        if "xy" in current_det and "hw" in current_det:
            detections.append(current_det)

        return detections

    @staticmethod
    def _remap_anyup_key(suffix):
        """Remap a PyTorch itok_upsampler weight key suffix to MLX AnyUp naming."""
        import re

        # ResBlock: block.<idx>.<param> → <mapped>
        # PyTorch Sequential indices: 0=GroupNorm, 1=SiLU, 2=Conv2d, 3=GroupNorm, 4=SiLU, 5=Conv2d
        BLOCK_MAP = {
            "0.weight": "norm1.weight",
            "0.bias": "norm1.bias",
            "2.weight": "conv1.weight",
            "3.weight": "norm2.weight",
            "3.bias": "norm2.bias",
            "5.weight": "conv2.weight",
        }

        # Encoder pattern: <enc>.<seq_idx>.block.<block_param> or <enc>.<seq_idx>.weight
        ENCODERS = [
            "image_encoder",
            "key_encoder",
            "query_encoder",
            "aggregation",
        ]
        for enc in ENCODERS:
            if not suffix.startswith(enc + "."):
                continue
            rest = suffix[len(enc) + 1 :]
            # First layer (Conv2d): "0.weight"
            if rest == "0.weight":
                return enc + ".conv.weight"
            # ResBlock layers: "<n>.block.<idx>.<param>"
            m = re.match(r"(\d+)\.block\.(.+)", rest)
            if m:
                block_idx = int(m.group(1)) - 1  # PyTorch 1-indexed → 0-indexed
                block_param = BLOCK_MAP.get(m.group(2))
                if block_param:
                    return f"{enc}.blocks.{block_idx}.{block_param}"
            # Shortcut: "<n>.shortcut.weight"
            m = re.match(r"(\d+)\.shortcut\.weight", rest)
            if m:
                block_idx = int(m.group(1)) - 1
                return f"{enc}.blocks.{block_idx}.shortcut.weight"

        # key_features_encoder (LFU + ResBlocks)
        if suffix.startswith("key_features_encoder."):
            rest = suffix[len("key_features_encoder.") :]
            if rest == "0.basis":
                return "key_features_encoder.lfu.basis"
            m = re.match(r"(\d+)\.block\.(.+)", rest)
            if m:
                block_idx = int(m.group(1)) - 1
                block_param = BLOCK_MAP.get(m.group(2))
                if block_param:
                    return f"key_features_encoder.blocks.{block_idx}.{block_param}"

        # cross_decode
        if suffix == "cross_decode.conv2d.weight":
            return "cross_decode.conv.weight"
        if suffix == "cross_decode.cross_attn.norm_q.weight":
            return "cross_decode.cross_attn.norm_q.weight"
        if suffix == "cross_decode.cross_attn.norm_k.weight":
            return "cross_decode.cross_attn.norm_k.weight"
        # in_proj_weight/bias handled separately (needs split)
        if suffix.startswith("cross_decode.cross_attn.attention.in_proj_"):
            return suffix  # handled by caller

        # rope
        if suffix == "rope.freqs":
            return "rope.freqs"

        return None

    def sanitize(self, weights):
        new_weights = {}
        anyup_attn_weight = None
        anyup_attn_bias = None

        for k, v in weights.items():
            # --- AnyUp weights ---
            if k.startswith("itok_upsampler."):
                suffix = k[len("itok_upsampler.") :]

                # Collect attention in_proj for splitting later
                if suffix == "cross_decode.cross_attn.attention.in_proj_weight":
                    anyup_attn_weight = v
                    continue
                if suffix == "cross_decode.cross_attn.attention.in_proj_bias":
                    anyup_attn_bias = v
                    continue

                new_suffix = self._remap_anyup_key(suffix)
                if new_suffix is None:
                    continue
                new_key = "itok_upsampler." + new_suffix

                # Transpose Conv2d weights: PyTorch [O,I,H,W] → MLX [O,H,W,I]
                if v.ndim == 4 and "norm" not in new_key and "basis" not in new_key:
                    v = v.transpose(0, 2, 3, 1)
                # LFU basis: PyTorch (out_ch, 1, k, k) → MLX (out_ch, k, k, 1)
                if "lfu.basis" in new_key:
                    v = v.transpose(0, 2, 3, 1)

                new_weights[new_key] = v
                continue

            # --- Standard backbone weights ---
            new_key = k

            if k.startswith("tok_embeddings."):
                new_key = k.replace(
                    "tok_embeddings.", "language_model.model.embed_tokens.", 1
                )
            elif k.startswith("img_projector."):
                new_key = k.replace(
                    "img_projector.", "language_model.model.img_projector.", 1
                )
            elif k.startswith("norm."):
                new_key = k.replace("norm.", "language_model.model.norm.", 1)
            elif k.startswith("output."):
                new_key = k.replace("output.", "language_model.lm_head.", 1)
            elif k == "freqs_cis_golden":
                new_key = "language_model.model.freqs_cis_golden"
            elif k.startswith("layers."):
                new_key = k.replace("layers.", "language_model.model.layers.", 1)
                new_key = new_key.replace(".attention.", ".self_attn.")
                new_key = new_key.replace(".feed_forward.", ".mlp.")

            if ".w13." in new_key:
                v = mx.concatenate([v[0::2], v[1::2]], axis=0)

            # Conv2d: transpose from PyTorch [O,I,H,W] to MLX [O,H,W,I]
            if "conv_segm.weight" in k:
                v = v.transpose(0, 2, 3, 1)

            new_weights[new_key] = v

        # Split AnyUp attention in_proj into q_proj and k_proj
        # in_proj_weight is (3*qk_dim, qk_dim) = (out, in), same layout as nn.Linear
        if anyup_attn_weight is not None:
            w_q, w_k, _ = mx.split(anyup_attn_weight, 3, axis=0)
            new_weights["itok_upsampler.cross_decode.cross_attn.q_proj.weight"] = w_q
            new_weights["itok_upsampler.cross_decode.cross_attn.k_proj.weight"] = w_k
        if anyup_attn_bias is not None:
            b_q, b_k, _ = mx.split(anyup_attn_bias, 3, axis=0)
            new_weights["itok_upsampler.cross_decode.cross_attn.q_proj.bias"] = b_q
            new_weights["itok_upsampler.cross_decode.cross_attn.k_proj.bias"] = b_k

        new_weights["language_model.model.cos_1d"] = self.language_model.model.cos_1d
        new_weights["language_model.model.sin_1d"] = self.language_model.model.sin_1d

        return new_weights
