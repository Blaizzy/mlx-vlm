"""SAM 3.1 multiplex video tracker.

Port of sam3/model/video_tracking_multiplex.py (inference subset; the training
branches sampling correction points from ground-truth masks are omitted).

Covers mask-as-output initialization (mask prompts bypass the SAM decoder),
memory-conditioned propagation (spatial mask memories, image features, object
pointer tokens), interactive point/mask refinement, and dynamic object
addition / reconditioning, plus the mask downsampler / memory encoder.

Weight keys: tracker_model.*
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..interpolate import resize_bilinear_nhwc
from ..sam3.position import PositionEmbeddingSine
from ..sam3.tracker import DownsampleConvBlock, MemoryFuser
from .config import TrackerConfig, TrackerMaskDecoderConfig
from .multiplex import MultiplexController, MultiplexState, MultiplexTrackerState
from .sam_components import (
    DecoupledMemoryAttention,
    MultiplexMaskDecoder,
    PositionalEmbedding,
    SAMPromptEncoder,
)

# A large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class MultiplexMaskDownSampler(nn.Module):
    """Mask downsampler for multiplex: 32 input channels (16 objects × 2).

    Weight keys: tracker_model.memory_encoder.mask_downsampler.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        first_ch = config.mask_downsampler_first_channels  # 16
        self.input_size = config.mask_downsampler_input_size  # 1152

        # Progressive: 32 -> 16 -> 64 -> 256 -> 1024 (from checkpoint)
        channels = [
            first_ch * 2,
            first_ch,
            first_ch * 4,
            first_ch * 16,
            first_ch * 64,
        ]
        conv_args = (
            config.mask_downsampler_kernel_size,
            config.mask_downsampler_stride,
            config.mask_downsampler_padding,
        )
        self.layers = [
            DownsampleConvBlock(in_ch, out_ch, *conv_args)
            for in_ch, out_ch in zip(channels, channels[1:])
        ]
        self.final_conv = nn.Conv2d(
            channels[-1], config.mask_downsampler_embed_dim, kernel_size=1, bias=True
        )

    def __call__(self, masks: mx.array) -> mx.array:
        """masks: (B, H, W, 2*multiplex_count) muxed mask channels."""
        if masks.shape[1:3] != (self.input_size, self.input_size):
            masks = resize_bilinear_nhwc(masks, (self.input_size, self.input_size))
        for layer in self.layers:
            masks = layer(masks)
        return self.final_conv(masks)


class MultiplexMemoryEncoder(nn.Module):
    """Memory encoder for SAM 3.1 multiplex.

    Weight keys: tracker_model.memory_encoder.*
    Note: SAM 3.1 uses dim=out_dim=256, so no separate output projection needed.
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        dim = config.memory_encoder_hidden_size

        self.mask_downsampler = MultiplexMaskDownSampler(config)
        self.memory_fuser = MemoryFuser(config)
        self.feature_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # Parameter-free sinusoidal 2D pos enc (256-dim output)
        self.position_encoding = PositionEmbeddingSine(dim // 2)

    def __call__(
        self, features: mx.array, masks: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """features: (1, H, W, D); masks: (B, H_m, W_m, 2M) muxed channels.

        Returns (memory, pos_enc), each (B, H, W, D).
        """
        fused = self.feature_projection(features) + self.mask_downsampler(masks)
        memory = self.memory_fuser(fused)
        return memory, self.position_encoding(memory)


class ObjectPointerMLP(nn.Module):
    """Projects SAM output tokens to object pointers.

    Weight keys: tracker_model.(interactive_)obj_ptr_proj.*
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(3)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)


def get_1d_sine_pe(pos_inds: mx.array, dim: int, temperature: float = 10000.0):
    """1D sine positional embedding as in the original Transformer paper."""
    pe_dim = dim // 2
    dim_t = mx.arange(pe_dim, dtype=mx.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds[..., None] / dim_t
    return mx.concatenate([mx.sin(pos_embed), mx.cos(pos_embed)], axis=-1)


def select_closest_cond_frames(
    frame_idx: int,
    cond_frame_outputs: Dict[int, dict],
    max_cond_frame_num: int,
) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Select up to `max_cond_frame_num` temporally closest conditioning frames."""
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return dict(cond_frame_outputs), {}

    assert max_cond_frame_num >= 2
    selected = {}
    # closest conditioning frame before and after `frame_idx`
    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected[idx_before] = cond_frame_outputs[idx_before]
    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected[idx_after] = cond_frame_outputs[idx_after]
    # fill up with the temporally closest remaining frames
    num_remain = max_cond_frame_num - len(selected)
    inds_remain = sorted(
        (t for t in cond_frame_outputs if t not in selected),
        key=lambda x: abs(x - frame_idx),
    )[:num_remain]
    selected.update({t: cond_frame_outputs[t] for t in inds_remain})
    unselected = {t: v for t, v in cond_frame_outputs.items() if t not in selected}
    return selected, unselected


def _resize_masks(
    masks: mx.array, out_h: int, out_w: int, antialias: bool = False
) -> mx.array:
    """Bilinear-resize the last two dims of (N, K, H, W) mask logits."""
    n, k, h, w = masks.shape
    if (h, w) == (out_h, out_w):
        return masks
    out = resize_bilinear_nhwc(
        masks.reshape(n * k, h, w, 1), (out_h, out_w), antialias=antialias
    )
    return out.reshape(n, k, out_h, out_w)


class MultiplexTrackerModel(nn.Module):
    """SAM 3.1 multiplex tracker (VideoTrackingMultiplex port).

    Weight keys: tracker_model.*
    """

    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.config = config
        d = config.memory_attention_hidden_size
        M = config.multiplex_count
        self.multiplex_count = M
        self.hidden_dim = d
        self.num_maskmem = config.num_maskmem
        self.max_cond_frames_in_attn = config.max_cond_frame_num
        self.max_obj_ptrs_in_encoder = config.max_object_pointers_in_encoder
        self.memory_temporal_stride_for_eval = config.memory_temporal_stride_for_eval
        self.object_score_logit_threshold = config.object_score_logit_threshold

        # Tracking behavior flags (SAM 3.1 build_sam3_multiplex_video_model)
        for name in (
            "use_obj_ptrs_in_encoder",
            "pred_obj_scores",
            "use_no_obj_ptr",
            "use_linear_no_obj_ptr",
            "fixed_no_obj_ptr",
            "add_output_suppression_embeddings",
            "condition_as_mask_input",
            "condition_as_mask_input_fg",
            "condition_as_mask_input_bg",
            "use_maskmem_tpos_v2",
            "save_image_features",
            "directly_add_no_mem_embed",
            "use_mask_input_as_output_without_sam",
            "apply_sigmoid_to_mask_logits_for_mem_enc",
            "sigmoid_scale_for_mem_enc",
            "sigmoid_bias_for_mem_enc",
            "multimask_output_in_sam",
            "multimask_output_for_tracking",
            "multimask_min_pt_num",
            "multimask_max_pt_num",
            "num_multimask_outputs",
        ):
            setattr(self, name, getattr(config, name))

        self.multiplex_controller = MultiplexController(M)

        # Interactive SAM components (point/box prompts, single object slot)
        self.interactive_sam_prompt_encoder = SAMPromptEncoder(
            config.prompt_encoder_config
        )
        interactive_cfg = TrackerMaskDecoderConfig(
            **{
                **config.mask_decoder_config.__dict__,
                "multiplex_count": 1,
                "num_multimask_outputs": config.num_multimask_outputs,
                "multimask_outputs_only": False,
                "dynamic_multimask_via_stability": False,
            }
        )
        self.interactive_sam_mask_decoder = MultiplexMaskDecoder(interactive_cfg)

        # Propagation SAM mask decoder (multiplex: 16 objects)
        self.sam_mask_decoder = MultiplexMaskDecoder(config.mask_decoder_config)

        # Memory components
        self.memory_attention = DecoupledMemoryAttention(config)
        self.memory_encoder = MultiplexMemoryEncoder(config)

        # Object pointer projections
        self.obj_ptr_proj = ObjectPointerMLP(d)
        self.interactive_obj_ptr_proj = ObjectPointerMLP(d)

        # Learned embeddings
        self.memory_temporal_positional_encoding = mx.zeros(
            (config.num_maskmem, 1, 1, d)
        )
        self.temporal_positional_encoding_projection_layer = nn.Linear(d, d)

        # Multiplex-specific embeddings
        self.output_valid_embed = mx.zeros((M, d))
        self.output_invalid_embed = mx.zeros((M, d))
        self.no_obj_embed_spatial = mx.zeros((M, d))
        self.no_obj_ptr_linear = nn.Linear(d, d)
        self.interactivity_no_mem_embed = mx.zeros((1, 1, d))

        # Image positional encoding
        self.image_pe_layer = PositionalEmbedding(d // 2)
        # Present in converted checkpoints for weight-load compatibility;
        # the reference model has no such module (it is unused)
        self.shared_image_embedding = PositionalEmbedding(d // 2)
        # Sinusoidal 2D pos enc for FPN features (parameter-free)
        self._pos_enc = PositionEmbeddingSine(d // 2)

        # Interactive mask downsample
        self.interactive_mask_downsample = nn.Conv2d(
            1, 1, kernel_size=4, stride=4, bias=True
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def init_state(
        self, num_objects: int, object_ids: Optional[List[int]] = None
    ) -> MultiplexTrackerState:
        """Create a fresh tracking session state for `num_objects` objects."""
        mux = self.multiplex_controller.get_state(
            num_objects, random=False, object_ids=object_ids
        )
        return MultiplexTrackerState(mux)

    # ------------------------------------------------------------------
    # Frame feature preparation (called by the Model wrapper)
    # ------------------------------------------------------------------

    def prepare_frame_features(
        self,
        interactive_fpn: Optional[List[mx.array]],
        propagation_fpn: Optional[List[mx.array]],
    ) -> dict:
        """Prepare per-frame backbone features (mirrors forward_image).

        Pre-applies the conv_s0/conv_s1 high-res projections so they are not
        recomputed on every decoder call, and computes the sine pos encs.
        Each fpn is [f0 (288), f1 (144), f2 (72)] or None.
        """

        def prep(fpn, decoder):
            return {
                "vision_feat": fpn[-1],
                "high_res": [decoder.conv_s0(fpn[0]), decoder.conv_s1(fpn[1])],
            }

        out = {}
        if interactive_fpn is not None:
            out["interactive"] = prep(
                interactive_fpn, self.interactive_sam_mask_decoder
            )
        if propagation_fpn is not None:
            out["propagation"] = prep(propagation_fpn, self.sam_mask_decoder)
            out["propagation"]["vision_pos"] = self._pos_enc(propagation_fpn[-1])
        return out

    # ------------------------------------------------------------------
    # SAM heads
    # ------------------------------------------------------------------

    def get_propagation_dense_pe(self) -> mx.array:
        """Dense positional encoding for the propagation mask decoder.

        Cached across frames (hidden from Module.parameters); keyed on the
        embedding weight so reloading weights invalidates the cache.
        """
        w = self.image_pe_layer.positional_embedding
        cached = self.__dict__.get("_prop_dense_pe")
        if cached is None or cached[0] is not w:
            side = self.config.mask_downsampler_input_size // 16  # 72
            pe = self.image_pe_layer((side, side))[None]  # (1, HW, D)
            object.__setattr__(self, "_prop_dense_pe", (w, pe))
            return pe
        return cached[1]

    def _get_interactive_pix_mem(self, vision_feat: mx.array) -> mx.array:
        """(1, H, W, D) -> (1, HW, D) with the no-memory embedding added."""
        assert self.directly_add_no_mem_embed
        B, H, W, D = vision_feat.shape
        return (vision_feat + self.interactivity_no_mem_embed).reshape(B, H * W, D)

    def _apply_no_obj_ptr(self, obj_ptr: mx.array, is_obj_appearing: mx.array):
        """Blend the learned no-object pointer into absent objects' pointers."""
        if not (self.pred_obj_scores and self.use_no_obj_ptr):
            return obj_ptr
        lam = is_obj_appearing.astype(mx.float32)
        if self.use_linear_no_obj_ptr:
            return lam * obj_ptr + (1 - lam) * self.no_obj_ptr_linear(obj_ptr)
        return lam * obj_ptr if self.fixed_no_obj_ptr else obj_ptr

    def _forward_sam_heads(
        self,
        backbone_features: mx.array,
        *,
        point_inputs: Optional[dict] = None,
        mask_inputs: Optional[mx.array] = None,
        interactive_high_res_features: Optional[List[mx.array]] = None,
        propagation_high_res_features: Optional[List[mx.array]] = None,
        multimask_output: bool = False,
        multiplex_state: MultiplexState,
    ) -> dict:
        """Forward the SAM prompt encoder + mask heads (interactive and/or
        multiplexed propagation path).

        Args:
            backbone_features: (B, HW, D) image features for the decoder
            point_inputs: point_coords (N, P, 2) absolute pixels, point_labels
                (N, P) with 1=positive, 0=negative, -1=padding
            mask_inputs: (N, 1, H_im, W_im) mask prompt
            multimask_output: output multiple candidate masks + IoU estimates
        Returns:
            dict with low/high-res (multi)masks, ious, object_score_logits
            (all (N, ...)) and obj_ptr (N, C)
        """
        is_interactive = point_inputs is not None or mask_inputs is not None

        if is_interactive:
            # Image-level, per-object interactive path
            assert interactive_high_res_features is not None

            if point_inputs is not None:
                sam_point_coords = point_inputs["point_coords"]
                sam_point_labels = point_inputs["point_labels"]
            else:
                # Pad with an empty point (label -1) when only masks are given
                sam_point_coords = mx.zeros((mask_inputs.shape[0], 1, 2))
                sam_point_labels = -mx.ones((mask_inputs.shape[0], 1), mx.int32)

            sam_mask_prompt = None
            if mask_inputs is not None:
                assert mask_inputs.ndim == 4
                # Downsize into the prompt encoder's mask input size (4x72=288)
                mask_size = (
                    4 * self.interactive_sam_prompt_encoder.image_embedding_size[0]
                )
                sam_mask_prompt = _resize_masks(
                    mask_inputs, mask_size, mask_size, antialias=True
                )

            # The prompt encoder pads with a not-a-point token when no boxes
            # are given (boxes are never used in the tracker)
            pad_coord = mx.zeros((sam_point_coords.shape[0], 1, 2))
            pad_label = -mx.ones((sam_point_labels.shape[0], 1), mx.int32)
            sam_point_coords = mx.concatenate([sam_point_coords, pad_coord], axis=1)
            sam_point_labels = mx.concatenate([sam_point_labels, pad_label], axis=1)

            # The MLX prompt encoder expects coords in embedding-grid units
            grid_scale = (
                self.interactive_sam_prompt_encoder.image_embedding_size[0]
                / self.config.image_size
            )
            sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
                points=(sam_point_coords * grid_scale, sam_point_labels),
                masks=(
                    None
                    if sam_mask_prompt is None
                    else sam_mask_prompt.transpose(0, 2, 3, 1)
                ),
            )

            out = self.interactive_sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=self.interactive_sam_prompt_encoder.get_dense_pe(),
                multimask_output=multimask_output,
                high_res_features=interactive_high_res_features,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            # Squeeze the singleton multiplex dim of the interactive decoder
            low_res_multimasks = out["masks"][:, 0]  # (N, K, h, w)
            ious = out["iou_pred"][:, 0]  # (N, K)
            sam_output_tokens = out["sam_tokens_out"][:, 0]  # (N, K', C)
            object_score_logits = out["object_score_logits"][:, 0]  # (N, 1)

        else:
            # Multiplexed propagation path
            assert propagation_high_res_features is not None

            output_merged_embed = None
            if self.add_output_suppression_embeddings:
                # Inform the mask decoder which slots hold valid objects
                valid_f = multiplex_state.get_valid_object_mask().astype(mx.float32)
                valid_f = valid_f[..., None]
                output_merged_embed = (
                    valid_f * self.output_valid_embed[None]
                    + (1 - valid_f) * self.output_invalid_embed[None]
                )

            out = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=self.get_propagation_dense_pe(),
                multimask_output=multimask_output,
                high_res_features=propagation_high_res_features,
                extra_per_object_embeddings=output_merged_embed,
            )
            low_res_multimasks = multiplex_state.demux(out["masks"])
            ious = multiplex_state.demux(out["iou_pred"])
            object_score_logits = multiplex_state.demux(out["object_score_logits"])
            sam_output_tokens = multiplex_state.demux(out["sam_tokens_out"])

        # The interactive and propagation paths converge here
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.object_score_logit_threshold
            # Hard choice between obj and no-obj for the spatial memories
            low_res_multimasks = mx.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        image_size = self.config.image_size
        high_res_multimasks = _resize_masks(low_res_multimasks, image_size, image_size)

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # Take the best mask prediction (highest estimated IoU)
            best_iou_inds = mx.argmax(ious, axis=-1)  # (N,)
            low_res_masks = mx.take_along_axis(
                low_res_multimasks, best_iou_inds[:, None, None, None], axis=1
            )
            high_res_masks = mx.take_along_axis(
                high_res_multimasks, best_iou_inds[:, None, None, None], axis=1
            )
            if sam_output_tokens.shape[1] > 1:
                sam_output_token = mx.take_along_axis(
                    sam_output_tokens, best_iou_inds[:, None, None], axis=1
                )[:, 0]
        else:
            low_res_masks = low_res_multimasks[:, 0:1]
            high_res_masks = high_res_multimasks[:, 0:1]

        # Object pointer from the SAM output token
        proj = self.interactive_obj_ptr_proj if is_interactive else self.obj_ptr_proj
        obj_ptr = proj(sam_output_token)
        if self.pred_obj_scores:
            obj_ptr = self._apply_no_obj_ptr(obj_ptr, is_obj_appearing)

        return {
            "low_res_multimasks": low_res_multimasks,
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
            "obj_ptr": obj_ptr,  # (N, C), in data space
        }

    def _use_mask_as_output(
        self,
        backbone_features: mx.array,
        high_res_features: List[mx.array],
        mask_inputs: mx.array,
        multiplex_state: MultiplexState,
    ) -> dict:
        """Turn binary `mask_inputs` directly into output mask logits (no SAM)."""
        # -10/+10 logits for neg/pos pixels (~0/1 after sigmoid)
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.astype(mx.float32)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        h, w = high_res_masks.shape[-2:]
        low_res_masks = _resize_masks(high_res_masks, h // 4, w // 4, antialias=True)

        # Produce an object pointer using the SAM decoder from the mask input
        mask_prompt = self.interactive_mask_downsample(
            mask_inputs_float.transpose(0, 2, 3, 1)
        ).transpose(0, 3, 1, 2)
        sam_outputs = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=mask_prompt,
            interactive_high_res_features=high_res_features,
            multiplex_state=multiplex_state,
        )

        # The mask input itself decides if the object appears
        is_obj_appearing = (
            (mask_inputs.reshape(mask_inputs.shape[0], -1) > 0.0)
            .any(axis=1)[:, None]
            .astype(mx.float32)
        )
        obj_ptr = self._apply_no_obj_ptr(sam_outputs["obj_ptr"], is_obj_appearing)

        return {
            "low_res_multimasks": low_res_masks,
            "high_res_multimasks": high_res_masks,
            # A dummy IoU prediction of all 1's under mask input
            "ious": mx.ones((mask_inputs.shape[0], 1)),
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": out_scale * is_obj_appearing + out_bias,
            "obj_ptr": obj_ptr,
        }

    # ------------------------------------------------------------------
    # Memory conditioning / encoding
    # ------------------------------------------------------------------

    def _get_tpos_enc(self, rel_pos_list: List[int], max_abs_pos: int) -> mx.array:
        """Temporal positional encoding for object pointers."""
        pos = mx.array(rel_pos_list, dtype=mx.float32) / (max_abs_pos - 1)
        pos_enc = get_1d_sine_pe(pos, dim=self.hidden_dim)
        return self.temporal_positional_encoding_projection_layer(pos_enc)

    def _prepare_memory_conditioned_features(
        self,
        *,
        frame_idx: int,
        current_vision_feat: mx.array,
        current_vision_pos: mx.array,
        output_dict: dict,
        num_frames: int,
        multiplex_state: MultiplexState,
        track_in_reverse: bool = False,
    ) -> mx.array:
        """Fuse current (1, H, W, C) features/pos encs with past memories.

        Returns (B=num_buckets, H, W, C) memory-conditioned features.
        """
        B = multiplex_state.num_buckets
        _, H, W, C = current_vision_feat.shape
        HW = H * W
        vision_feat = mx.broadcast_to(current_vision_feat.reshape(1, HW, C), (B, HW, C))
        src_pos = mx.broadcast_to(current_vision_pos.reshape(1, HW, C), (B, HW, C))

        # Gather spatial mask memories, their pos encs, and image features
        to_cat_prompt, to_cat_prompt_pos = [], []
        to_cat_image, to_cat_image_pos = [], []

        selected_cond, unselected_cond = select_closest_cond_frames(
            frame_idx,
            output_dict["cond_frame_outputs"],
            self.max_cond_frames_in_attn,
        )

        tpos_sign_mul = -1 if track_in_reverse else 1
        t_pos_and_prevs = [
            ((frame_idx - t) * tpos_sign_mul, out, True)
            for t, out in selected_cond.items()
        ]

        # Last (num_maskmem - 1) frames as non-conditioning memory
        r = self.memory_temporal_stride_for_eval
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if t_rel == 1:
                # the frame immediately before/after this frame
                prev_frame_idx = (
                    frame_idx - t_rel if not track_in_reverse else frame_idx + t_rel
                )
            elif not track_in_reverse:
                prev_frame_idx = ((frame_idx - 2) // r) * r - (t_rel - 2) * r
            else:
                prev_frame_idx = -(-(frame_idx + 2) // r) * r + (t_rel - 2) * r
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx)
            if out is None:
                out = unselected_cond.get(prev_frame_idx)
            t_pos_and_prevs.append((t_pos, out, False))

        for t_pos, prev, is_cond in t_pos_and_prevs:
            if prev is None or prev.get("maskmem_features") is None:
                continue
            to_cat_prompt.append(prev["maskmem_features"].reshape(B, HW, C))
            mem_pos = prev["maskmem_pos_enc"].reshape(B, HW, C)

            if self.use_maskmem_tpos_v2:
                # out-of-range t_pos maps to the last ("out-of-range") slot
                t_idx = (
                    self.num_maskmem - t_pos - 1
                    if 0 < t_pos < self.num_maskmem
                    else self.num_maskmem - 1
                )
            else:
                t_idx = self.num_maskmem - (0 if is_cond else t_pos) - 1
            tpos = self.memory_temporal_positional_encoding[t_idx]
            to_cat_prompt_pos.append(mem_pos + tpos.reshape(1, 1, C))

            if self.save_image_features:
                to_cat_image.append(prev["image_features"].reshape(1, HW, C))
                to_cat_image_pos.append(
                    prev["image_pos_enc"].reshape(1, HW, C) + tpos.reshape(1, 1, C)
                )

        # Object pointers from past frames
        num_obj_ptr_tokens = 0
        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
            pos_and_outs = [
                (abs(frame_idx - t), out) for t, out in selected_cond.items()
            ]
            for t_diff in range(1, max_obj_ptrs):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or t >= num_frames:
                    break
                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond.get(t)
                )
                if out is not None:
                    pos_and_outs.append((t_diff, out))

            filtered = [(p, o) for p, o in pos_and_outs if "obj_ptr" in o]
            if filtered:
                pos_list, out_list = zip(*filtered)
                # muxed ptrs per frame: (B, M, C) -> (B, T*M, C)
                obj_ptrs = mx.concatenate([out["obj_ptr"] for out in out_list], axis=1)
                obj_pos = self._get_tpos_enc(
                    list(pos_list), max_abs_pos=max_obj_ptrs
                )  # (T, C)
                # Each frame contributes multiplex_count pointers
                obj_pos = mx.repeat(obj_pos, self.multiplex_count, axis=0)
                obj_pos = mx.broadcast_to(obj_pos[None], (B, *obj_pos.shape))

                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_pos.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[1]

        if not to_cat_prompt:
            # No available memories; propagate from current features only
            return vision_feat.reshape(B, H, W, C)

        memory = mx.concatenate(to_cat_prompt, axis=1)
        memory_pos = mx.concatenate(to_cat_prompt_pos, axis=1)

        if self.save_image_features:
            if not to_cat_image:
                return vision_feat.reshape(B, H, W, C)
            memory_image = mx.concatenate(to_cat_image, axis=1)
            memory_image_pos = mx.concatenate(to_cat_image_pos, axis=1)
        else:
            memory_image, memory_image_pos = memory, memory_pos

        pix_feat_with_mem = self.memory_attention(
            image=current_vision_feat.reshape(1, HW, C),
            src=vision_feat,
            memory_image=memory_image,
            memory=memory,
            src_pos=src_pos,
            memory_pos=memory_pos,
            memory_image_pos=memory_image_pos,
            num_k_exclude_rope=num_obj_ptr_tokens,
        )
        return pix_feat_with_mem.reshape(B, H, W, C)

    def _encode_new_memory(
        self,
        *,
        current_vision_feat: mx.array,
        pred_masks_high_res: mx.array,
        object_score_logits: mx.array,
        conditioning_objects,
        multiplex_state: MultiplexState,
    ) -> Tuple[mx.array, mx.array]:
        """Encode the current frame's predictions into a memory feature.

        pred_masks_high_res: (N, 1, H_im, W_im); object_score_logits: (N, 1).
        Returns (maskmem_features, maskmem_pos_enc), each (B, H, W, C).
        """
        mask_for_mem = pred_masks_high_res
        if self.apply_sigmoid_to_mask_logits_for_mem_enc:
            mask_for_mem = (
                mx.sigmoid(pred_masks_high_res) * self.sigmoid_scale_for_mem_enc
                + self.sigmoid_bias_for_mem_enc
            )

        # (the reference also computes unconditioned objects here, only used by
        # the object-conditional embeddings that SAM 3.1 disables)
        conditioning_objects = sorted(conditioning_objects or ())

        # (N, 1, H, W) -> mux -> (B, M, H, W)
        mux_mask = multiplex_state.mux(mask_for_mem[:, 0])

        if self.condition_as_mask_input:
            # Extra per-object channel marking conditioning objects
            cond_values = mx.full(
                (mask_for_mem.shape[0],), self.condition_as_mask_input_bg, mx.float32
            )
            if conditioning_objects:
                cond_values[conditioning_objects] = self.condition_as_mask_input_fg
            embedded = mx.broadcast_to(
                cond_values[:, None, None], mask_for_mem[:, 0].shape
            )
            mux_mask = mx.concatenate([mux_mask, multiplex_state.mux(embedded)], axis=1)

        # (B, 2M, H, W) -> (B, H, W, 2M) channel-last for the convs
        mux_mask = mux_mask.transpose(0, 2, 3, 1)
        maskmem_features, maskmem_pos_enc = self.memory_encoder(
            current_vision_feat, mux_mask
        )

        # Add a projected embedding for each empty object slot
        obj_logits = object_score_logits
        num_missing = multiplex_state.total_valid_entries - obj_logits.shape[0]
        if num_missing > 0:
            pad = mx.zeros((num_missing, *obj_logits.shape[1:]))
            obj_logits = mx.concatenate([obj_logits, pad], axis=0)
        elif num_missing < 0:
            obj_logits = obj_logits[: multiplex_state.total_valid_entries]
        appearing = multiplex_state.mux(obj_logits)
        is_obj_appearing = (appearing > self.object_score_logit_threshold).astype(
            mx.float32
        )  # (B, M, 1)
        no_obj_embed = ((1 - is_obj_appearing) * self.no_obj_embed_spatial[None]).sum(
            axis=1
        )  # (B, C)
        maskmem_features = maskmem_features + no_obj_embed[:, None, None, :]

        return maskmem_features, maskmem_pos_enc

    # ------------------------------------------------------------------
    # Track step
    # ------------------------------------------------------------------

    def _use_multimask(self, is_init_cond_frame: bool, point_inputs) -> bool:
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
            and self.num_multimask_outputs > 0
        )

    def track_step(
        self,
        state: MultiplexTrackerState,
        *,
        frame_idx: int,
        is_init_cond_frame: bool,
        frame_features: dict,
        point_inputs: Optional[dict] = None,
        mask_inputs: Optional[mx.array] = None,
        num_frames: Optional[int] = None,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Optional[mx.array] = None,
        objects_to_interact: Optional[List[int]] = None,
        new_object_masks: Optional[mx.array] = None,
        new_object_idxs: Optional[List[int]] = None,
        new_object_ids: Optional[List[int]] = None,
        are_new_masks_from_pts: bool = False,
    ) -> dict:
        """Run one tracking step on a frame.

        Four modes, selected from the inputs: mask-as-output (mask_inputs
        given), propagation-only (no prompts), interaction-only (points on a
        conditioning frame, or refinement with prev_sam_mask_logits), and
        propagation-and-interaction (points on a non-conditioning frame).

        Args:
            frame_features: dict from prepare_frame_features
            num_frames: total video length (for object-pointer range limiting)
        Returns:
            the frame's StageOutput dict (also stored into `state`)
        """
        if num_frames is None:
            num_frames = frame_idx + 1

        current_out, aux_out = self._track_step_aux(
            state,
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            frame_features=frame_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=(run_mem_encoder and new_object_masks is None),
            prev_sam_mask_logits=prev_sam_mask_logits,
            objects_to_interact=objects_to_interact,
            need_aux_output=(new_object_masks is not None),
        )

        if new_object_masks is not None:
            assert new_object_idxs is not None
            self.add_new_masks_to_existing_state(
                interactive_pix_feat=aux_out["interactive_pix_feat"],
                interactive_high_res_features=aux_out["interactive_high_res_features"],
                propagation_vision_feat=aux_out["propagation_vision_feat"],
                new_masks=new_object_masks,
                obj_idxs_in_mask=new_object_idxs,
                obj_ids_in_mask=new_object_ids,
                prev_output=current_out,
                state=state,
                add_mask_to_memory=run_mem_encoder,
                are_masks_from_pts=are_new_masks_from_pts,
            )

        if is_init_cond_frame:
            state.cond_frame_outputs[frame_idx] = current_out
        else:
            state.non_cond_frame_outputs[frame_idx] = current_out

        # Prune stale non-conditioning outputs that can no longer be
        # referenced by the memory attention (bounds session memory)
        max_lookback = max(
            self.max_obj_ptrs_in_encoder,
            2 + (self.num_maskmem - 2) * self.memory_temporal_stride_for_eval,
        )
        cutoff = frame_idx - max_lookback
        for t in [t for t in state.non_cond_frame_outputs if t < cutoff]:
            del state.non_cond_frame_outputs[t]

        return current_out

    def _track_step_aux(
        self,
        state: MultiplexTrackerState,
        *,
        frame_idx: int,
        is_init_cond_frame: bool,
        frame_features: dict,
        point_inputs: Optional[dict],
        mask_inputs: Optional[mx.array],
        num_frames: int,
        track_in_reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: Optional[mx.array],
        objects_to_interact: Optional[List[int]],
        need_aux_output: bool,
    ) -> Tuple[dict, dict]:
        multiplex_state = state.multiplex_state
        interactive = frame_features.get("interactive")
        propagation = frame_features.get("propagation")

        current_out = {
            "conditioning_objects": set(),
            "point_inputs": point_inputs,
            "mask_inputs": mask_inputs,
        }

        # Determine the tracking mode
        if mask_inputs is not None:
            mode = "mask_as_output"
        elif point_inputs is None:
            mode = "propagation_only"
        elif prev_sam_mask_logits is not None or is_init_cond_frame:
            mode = "interaction_only"
        elif objects_to_interact is not None:
            mode = "propagation_and_interaction"
        else:
            raise ValueError(
                "Unable to determine tracking mode: "
                f"mask_inputs={mask_inputs is not None}, "
                f"point_inputs={point_inputs is not None}, "
                f"prev_sam_mask_logits={prev_sam_mask_logits is not None}, "
                f"{objects_to_interact=}, {is_init_cond_frame=}"
            )

        if mode in ("interaction_only", "propagation_and_interaction"):
            assert interactive is not None
        if mode in ("propagation_only", "propagation_and_interaction"):
            assert propagation is not None

        interactive_pix_feat = None
        if mode == "mask_as_output":
            assert self.use_mask_input_as_output_without_sam
            assert interactive is not None
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive["vision_feat"]
            )
            sam_outputs = self._use_mask_as_output(
                backbone_features=interactive_pix_feat,
                high_res_features=interactive["high_res"],
                mask_inputs=mask_inputs,
                multiplex_state=multiplex_state,
            )
            current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
        else:
            propagation_out = None
            if mode in ("propagation_only", "propagation_and_interaction"):
                pix_feat_with_mem = self._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    current_vision_feat=propagation["vision_feat"],
                    current_vision_pos=propagation["vision_pos"],
                    output_dict=state.output_dict,
                    num_frames=num_frames,
                    multiplex_state=multiplex_state,
                    track_in_reverse=track_in_reverse,
                )
                B, H, W, C = pix_feat_with_mem.shape
                multimask_output = self._use_multimask(is_init_cond_frame, None)
                propagation_out = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem.reshape(B, H * W, C),
                    propagation_high_res_features=propagation["high_res"],
                    multimask_output=multimask_output,
                    multiplex_state=multiplex_state,
                )

            interaction_out = None
            if mode in ("interaction_only", "propagation_and_interaction"):
                interactive_pix_feat = self._get_interactive_pix_mem(
                    interactive["vision_feat"]
                )
                assert mask_inputs is None and point_inputs is not None
                if prev_sam_mask_logits is not None:
                    assert objects_to_interact is not None
                    assert mode != "propagation_and_interaction"
                    mask_inputs = prev_sam_mask_logits[objects_to_interact]
                elif mode == "propagation_and_interaction":
                    # Use the propagated masks as mask input
                    mask_inputs = propagation_out["low_res_masks"][objects_to_interact]

                if objects_to_interact is not None:
                    assert (
                        point_inputs["point_coords"].shape[0]
                        == point_inputs["point_labels"].shape[0]
                        == len(objects_to_interact)
                    )

                multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
                interaction_out = self._forward_sam_heads(
                    backbone_features=interactive_pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    interactive_high_res_features=interactive["high_res"],
                    multimask_output=multimask_output,
                    multiplex_state=multiplex_state,
                )
                current_out["conditioning_objects"].update(
                    objects_to_interact
                    if objects_to_interact is not None
                    else multiplex_state.get_all_valid_object_idx()
                )

            if propagation_out is None:
                sam_outputs = interaction_out
            elif interaction_out is None:
                sam_outputs = propagation_out
            else:
                # Merge: replace the interacted objects in the propagated output
                for k in (
                    "low_res_multimasks",
                    "high_res_multimasks",
                    "low_res_masks",
                    "high_res_masks",
                    "ious",
                    "object_score_logits",
                    "obj_ptr",
                ):
                    propagation_out[k][objects_to_interact] = interaction_out[k]
                sam_outputs = propagation_out

        current_out["pred_masks"] = sam_outputs["low_res_masks"]
        current_out["pred_masks_high_res"] = sam_outputs["high_res_masks"]
        current_out["object_score_logits"] = sam_outputs["object_score_logits"]
        if self.use_obj_ptrs_in_encoder:
            # Object pointers are stored in the multiplex space
            current_out["obj_ptr"] = multiplex_state.mux(sam_outputs["obj_ptr"])

        # Encode the predicted masks into a new memory for future frames
        if run_mem_encoder and self.num_maskmem > 0:
            features, pos_enc = self._encode_new_memory(
                current_vision_feat=propagation["vision_feat"],
                pred_masks_high_res=current_out["pred_masks_high_res"],
                object_score_logits=current_out["object_score_logits"],
                conditioning_objects=current_out["conditioning_objects"],
                multiplex_state=multiplex_state,
            )
            current_out["maskmem_features"] = features
            current_out["maskmem_pos_enc"] = pos_enc

        if self.save_image_features:
            current_out["image_features"] = propagation["vision_feat"]
            current_out["image_pos_enc"] = propagation["vision_pos"]

        aux_output = {}
        if need_aux_output:
            if interactive_pix_feat is None:
                interactive_pix_feat = self._get_interactive_pix_mem(
                    interactive["vision_feat"]
                )
            aux_output["interactive_pix_feat"] = interactive_pix_feat
            aux_output["interactive_high_res_features"] = interactive["high_res"]
            aux_output["propagation_vision_feat"] = (
                propagation["vision_feat"] if propagation is not None else None
            )

        return current_out, aux_output

    # ------------------------------------------------------------------
    # Dynamic object management
    # ------------------------------------------------------------------

    def _merge_mask_output(
        self,
        prev_output: dict,
        mask_output: dict,
        multiplex_state: MultiplexState,
        *,
        conditioned: List[int],
        obj_idxs: Optional[List[int]] = None,
        existing_pointers: Optional[mx.array] = None,
    ):
        """Merge mask-encoded objects into `prev_output` (in place).

        Appends the new rows, or replaces the `obj_idxs` rows when given
        (reconditioning); `conditioned` objects condition on this frame.
        """
        h, w = prev_output["pred_masks"].shape[-2:]
        new_values = (
            _resize_masks(mask_output["low_res_masks"], h, w, antialias=True),
            mask_output["high_res_masks"],
            mask_output["object_score_logits"],
        )
        for key, val in zip(
            ("pred_masks", "pred_masks_high_res", "object_score_logits"), new_values
        ):
            if key not in prev_output:
                continue
            if obj_idxs is None:
                prev_output[key] = mx.concatenate([prev_output[key], val], axis=0)
            else:
                prev_output[key][obj_idxs] = val

        if self.use_obj_ptrs_in_encoder:
            if obj_idxs is None:
                pointers = mx.concatenate(
                    [existing_pointers, mask_output["obj_ptr"]], axis=0
                )
            else:
                pointers = multiplex_state.demux(prev_output["obj_ptr"])
                pointers[obj_idxs] = mask_output["obj_ptr"]
            prev_output["obj_ptr"] = multiplex_state.mux(pointers)

        prev_output["conditioning_objects"].update(conditioned)

    def _reencode_memory(
        self,
        prev_output: dict,
        propagation_vision_feat: Optional[mx.array],
        multiplex_state: MultiplexState,
    ):
        """Re-encode the spatial memory from the merged predictions."""
        features, pos_enc = self._encode_new_memory(
            current_vision_feat=propagation_vision_feat,
            pred_masks_high_res=prev_output["pred_masks_high_res"],
            object_score_logits=prev_output["object_score_logits"],
            conditioning_objects=prev_output["conditioning_objects"],
            multiplex_state=multiplex_state,
        )
        prev_output["maskmem_features"] = features
        prev_output["maskmem_pos_enc"] = pos_enc

    def add_new_masks_to_existing_state(
        self,
        *,
        interactive_pix_feat: mx.array,
        interactive_high_res_features: List[mx.array],
        propagation_vision_feat: Optional[mx.array],
        new_masks: mx.array,
        obj_idxs_in_mask: List[int],
        obj_ids_in_mask: Optional[List[int]],
        prev_output: dict,
        state: MultiplexTrackerState,
        add_mask_to_memory: bool = True,
        are_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ):
        """Append new objects to an existing output/multiplex state (in-place)."""
        assert self.use_mask_input_as_output_without_sam
        multiplex_state = state.multiplex_state
        num_new_objects = new_masks.shape[0]
        assert num_new_objects == len(obj_idxs_in_mask)

        existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        # Step 1: extend the multiplex state
        new_object_idx = multiplex_state.find_next_batch_of_available_indices(
            num_objects=num_new_objects,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )
        multiplex_state.add_objects(
            object_indices=new_object_idx,
            object_ids=obj_ids_in_mask,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )

        # Step 2: encode the incoming masks
        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
        )

        # Step 3: match the high-res resolutions, then append the new objects
        if "pred_masks_high_res" in prev_output:
            res = mask_output["high_res_masks"].shape[-1]
            prev_output["pred_masks_high_res"] = _resize_masks(
                prev_output["pred_masks_high_res"], res, res
            )
        self._merge_mask_output(
            prev_output,
            mask_output,
            multiplex_state,
            conditioned=new_object_idx,
            existing_pointers=existing_pointers,
        )

        # Step 4: re-encode the spatial memory
        if add_mask_to_memory:
            assert (
                prev_output["pred_masks_high_res"].shape[0]
                == multiplex_state.total_valid_entries
            )
            self._reencode_memory(prev_output, propagation_vision_feat, multiplex_state)

    def recondition_masks_in_existing_state(
        self,
        *,
        interactive_pix_feat: mx.array,
        interactive_high_res_features: List[mx.array],
        propagation_vision_feat: Optional[mx.array],
        new_masks: mx.array,
        obj_idxs_in_mask: List[int],
        obj_ids_in_mask: Optional[List[int]],
        prev_output: dict,
        state: MultiplexTrackerState,
        add_mask_to_memory: bool = True,
    ):
        """Recondition existing objects with new masks (in-place)."""
        assert self.use_mask_input_as_output_without_sam
        multiplex_state = state.multiplex_state
        assert new_masks.shape[0] == len(obj_idxs_in_mask)

        # Step 1: encode the incoming masks
        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
        )

        # Step 2: replace the reconditioned objects in the existing state
        self._merge_mask_output(
            prev_output,
            mask_output,
            multiplex_state,
            conditioned=obj_idxs_in_mask,
            obj_idxs=obj_idxs_in_mask,
        )

        # Step 3: re-encode the spatial memory
        if add_mask_to_memory:
            self._reencode_memory(prev_output, propagation_vision_feat, multiplex_state)

    # ------------------------------------------------------------------
    # Convenience session API
    # ------------------------------------------------------------------

    def add_mask_prompt(
        self,
        state: MultiplexTrackerState,
        frame_idx: int,
        frame_features: dict,
        masks: mx.array,
        object_ids: Optional[List[int]] = None,
    ) -> dict:
        """Initialize (or extend) a session with mask prompts on a frame.

        masks: (N, H_im, W_im) binary/float masks at image resolution.
        Returns the frame's output dict.
        """
        if masks.ndim == 3:
            masks = masks[:, None]

        prev_output = state.cond_frame_outputs.get(
            frame_idx, state.non_cond_frame_outputs.get(frame_idx)
        )
        if prev_output is None:
            # Fresh frame: the masks are the conditioning input (mask-as-output)
            return self.track_step(
                state,
                frame_idx=frame_idx,
                is_init_cond_frame=True,
                frame_features=frame_features,
                mask_inputs=masks,
                num_frames=frame_idx + 1,
            )

        # Frame already tracked: merge the masks as new objects
        new_idxs = state.multiplex_state.find_next_batch_of_available_indices(
            masks.shape[0], allow_new_buckets=True
        )
        self.add_new_masks_to_existing_state(
            interactive_pix_feat=self._get_interactive_pix_mem(
                frame_features["interactive"]["vision_feat"]
            ),
            interactive_high_res_features=frame_features["interactive"]["high_res"],
            propagation_vision_feat=frame_features["propagation"]["vision_feat"],
            new_masks=masks,
            obj_idxs_in_mask=new_idxs,
            obj_ids_in_mask=object_ids,
            prev_output=prev_output,
            state=state,
        )
        return prev_output

    def propagate(
        self,
        state: MultiplexTrackerState,
        frame_idx: int,
        frame_features: dict,
        num_frames: Optional[int] = None,
        run_mem_encoder: bool = True,
    ) -> dict:
        """Propagate all tracked objects to a new frame.

        Returns the frame's output dict; per-object masks are
        out["pred_masks"] (N, 1, h, w) and out["pred_masks_high_res"].
        """
        return self.track_step(
            state,
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            frame_features=frame_features,
            num_frames=num_frames,
            run_mem_encoder=run_mem_encoder,
        )
