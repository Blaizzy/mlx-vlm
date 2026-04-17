"""Convert SAM 3.1 Meta checkpoint to MLX safetensors format.

Usage:
    python -m mlx_vlm.models.sam3_1.convert_weights --output /path/to/output
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file


def load_checkpoint():
    """Download and load the Meta SAM 3.1 checkpoint."""
    pt_path = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt")
    return torch.load(pt_path, map_location="cpu", weights_only=True)


def split_qkv(weight, bias=None):
    """Split fused QKV weight (3*D, D) into separate Q, K, V."""
    D = weight.shape[0] // 3
    q_w, k_w, v_w = weight[:D], weight[D : 2 * D], weight[2 * D :]
    if bias is not None:
        q_b, k_b, v_b = bias[:D], bias[D : 2 * D], bias[2 * D :]
        return (q_w, k_w, v_w), (q_b, k_b, v_b)
    return (q_w, k_w, v_w), None


# ---------------------------------------------------------------------------
# FPN neck sub-key mapping
# ---------------------------------------------------------------------------


def _map_fpn_conv_key(suffix):
    """Map FPN conv sub-keys: dconv_2x2_0 -> scale_layers.0, etc."""
    # dconv_2x2_0 -> scale_layers.0  (ConvTranspose2d, first upsample)
    # dconv_2x2_1 -> scale_layers.2  (ConvTranspose2d, second upsample for 4x)
    # dconv_2x2   -> scale_layers.0  (ConvTranspose2d, single 2x upsample)
    # conv_1x1    -> proj1
    # conv_3x3    -> proj2
    suffix = suffix.replace("dconv_2x2_0", "scale_layers.0")
    suffix = suffix.replace("dconv_2x2_1", "scale_layers.2")
    suffix = suffix.replace("dconv_2x2", "scale_layers.0")
    suffix = suffix.replace("conv_1x1", "proj1")
    suffix = suffix.replace("conv_3x3", "proj2")
    return suffix


# ---------------------------------------------------------------------------
# Key groups that contain fused QKV that need splitting
# ---------------------------------------------------------------------------

# Patterns where we expect fused QKV (in_proj_weight/bias or qkv.weight/bias)
QKV_FUSED_PATTERNS = [
    # ViT attention qkv
    re.compile(r".*\.attn\.qkv\.(weight|bias)$"),
    # CLIP text encoder in_proj
    re.compile(r".*\.attn\.in_proj_(weight|bias)$"),
    # DETR encoder/decoder self_attn, cross_attn
    re.compile(r".*\.self_attn\.in_proj_(weight|bias)$"),
    re.compile(r".*\.cross_attn\.in_proj_(weight|bias)$"),
    re.compile(r".*\.cross_attn_image\.in_proj_(weight|bias)$"),
    re.compile(r".*\.ca_text\.in_proj_(weight|bias)$"),
    # Geometry encoder
    # Segmentation head cross_attend_prompt
    re.compile(r".*\.cross_attend_prompt\.in_proj_(weight|bias)$"),
    # Tracker memory attention
    re.compile(r".*\.cross_attention\.in_proj_(weight|bias)$"),
    re.compile(r".*self_attn\.in_proj_(weight|bias)$"),
]

# Keys that are ConvTranspose2d (need different transpose order)
CONV_TRANSPOSE_PATTERNS = [
    re.compile(r".*dconv_2x2.*\.(weight|bias)$"),
    re.compile(r".*\.pixel_decoder\.dconv.*\.(weight|bias)$"),
]


def is_qkv_fused(key):
    """Check if a key is a fused QKV parameter."""
    for pat in QKV_FUSED_PATTERNS:
        if pat.match(key):
            return True
    return False


def is_conv_transpose(key):
    """Check if a key is a ConvTranspose2d weight."""
    for pat in CONV_TRANSPOSE_PATTERNS:
        if pat.match(key):
            return True
    return False


# ---------------------------------------------------------------------------
# Main key conversion
# ---------------------------------------------------------------------------


def convert_key(old_key):
    """Map a Meta checkpoint key to MLX format.

    Returns (new_key, skip) where skip=True means the key should be dropped.
    For QKV keys, returns None (handled separately).
    """
    key = old_key

    # ------------------------------------------------------------------
    # Skip buffers
    # ------------------------------------------------------------------
    if "freqs_cis" in key:
        return None, True

    # ==================================================================
    # DETECTOR keys
    # ==================================================================
    if key.startswith("detector."):
        key = key[len("detector.") :]  # strip "detector."

        # --- Vision backbone trunk ---
        # ViT blocks
        m = re.match(r"backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.(.+)", key)
        if m:
            block_idx = m.group(1)
            rest = m.group(2)
            prefix = f"detector_model.vision_encoder.backbone.layers.{block_idx}"

            # attn sub-keys (excluding qkv which is handled separately)
            if rest.startswith("attn."):
                attn_rest = rest[len("attn.") :]
                # qkv is handled by QKV splitter
                if attn_rest.startswith("qkv."):
                    return None, False  # signal caller to handle QKV
                # Meta uses attn.proj for output projection -> out_proj
                # But don't double-prefix if already out_proj (from probe keys)
                if attn_rest.startswith("proj.") and not attn_rest.startswith("proj_"):
                    attn_rest = "out_proj." + attn_rest[len("proj.") :]
                return f"{prefix}.attention.{attn_rest}", False

            if rest.startswith("norm1."):
                return f"{prefix}.layer_norm1.{rest[len('norm1.'):]}", False
            if rest.startswith("norm2."):
                return f"{prefix}.layer_norm2.{rest[len('norm2.'):]}", False
            if rest.startswith("mlp."):
                return f"{prefix}.mlp.{rest[len('mlp.'):]}", False

            # ls1, ls2 (layer scale)
            if rest.startswith("ls1."):
                return f"{prefix}.layer_scale1.{rest[len('ls1.'):]}", False
            if rest.startswith("ls2."):
                return f"{prefix}.layer_scale2.{rest[len('ls2.'):]}", False

            return f"{prefix}.{rest}", False

        # ln_pre -> layer_norm
        if key.startswith("backbone.vision_backbone.trunk.ln_pre."):
            rest = key[len("backbone.vision_backbone.trunk.ln_pre.") :]
            return f"detector_model.vision_encoder.backbone.layer_norm.{rest}", False

        # patch_embed.proj -> embeddings.patch_embeddings.projection
        if key.startswith("backbone.vision_backbone.trunk.patch_embed.proj."):
            rest = key[len("backbone.vision_backbone.trunk.patch_embed.proj.") :]
            return (
                f"detector_model.vision_encoder.backbone.embeddings.patch_embeddings.projection.{rest}",
                False,
            )

        # pos_embed -> position_embeddings
        if key == "backbone.vision_backbone.trunk.pos_embed":
            return (
                "detector_model.vision_encoder.backbone.embeddings.position_embeddings",
                False,
            )

        # Other trunk keys (pos_embed_window, etc.)
        if key.startswith("backbone.vision_backbone.trunk."):
            rest = key[len("backbone.vision_backbone.trunk.") :]
            return f"detector_model.vision_encoder.backbone.{rest}", False

        # --- FPN neck convs ---
        for conv_type in ["convs", "interactive_convs", "propagation_convs"]:
            if key.startswith(f"backbone.vision_backbone.{conv_type}."):
                rest = key[len(f"backbone.vision_backbone.{conv_type}.") :]
                rest = _map_fpn_conv_key(rest)
                return f"detector_model.vision_encoder.neck.{conv_type}.{rest}", False

        # --- Language backbone (CLIP text encoder) ---
        if key.startswith("backbone.language_backbone."):
            lk = key[len("backbone.language_backbone.") :]

            # text_projection (can be either top-level or under encoder.)
            if lk == "text_projection" or lk == "encoder.text_projection":
                return "detector_model.text_encoder.text_projection.weight", False

            # encoder.ln_final -> text_model.final_layer_norm
            if lk.startswith("encoder.ln_final."):
                rest = lk[len("encoder.ln_final.") :]
                return (
                    f"detector_model.text_encoder.text_model.final_layer_norm.{rest}",
                    False,
                )

            # encoder.token_embedding -> text_model.embeddings.token_embedding
            if lk.startswith("encoder.token_embedding."):
                rest = lk[len("encoder.token_embedding.") :]
                return (
                    f"detector_model.text_encoder.text_model.embeddings.token_embedding.{rest}",
                    False,
                )

            # encoder.positional_embedding -> text_model.embeddings.position_embedding.weight
            if lk == "encoder.positional_embedding":
                return (
                    "detector_model.text_encoder.text_model.embeddings.position_embedding.weight",
                    False,
                )

            # encoder.transformer.resblocks.{N}
            m2 = re.match(r"encoder\.transformer\.resblocks\.(\d+)\.(.+)", lk)
            if m2:
                layer_idx = m2.group(1)
                rest = m2.group(2)
                prefix = (
                    f"detector_model.text_encoder.text_model.encoder.layers.{layer_idx}"
                )

                # attn.in_proj_weight/bias -> QKV split (handled separately)
                if rest.startswith("attn.in_proj_"):
                    return None, False

                if rest.startswith("attn.out_proj."):
                    attn_rest = rest[len("attn.out_proj.") :]
                    return f"{prefix}.self_attn.out_proj.{attn_rest}", False

                if rest.startswith("ln_1."):
                    return f"{prefix}.layer_norm1.{rest[len('ln_1.'):]}", False
                if rest.startswith("ln_2."):
                    return f"{prefix}.layer_norm2.{rest[len('ln_2.'):]}", False

                if rest.startswith("mlp.c_fc."):
                    return f"{prefix}.mlp.fc1.{rest[len('mlp.c_fc.'):]}", False
                if rest.startswith("mlp.c_proj."):
                    return f"{prefix}.mlp.fc2.{rest[len('mlp.c_proj.'):]}", False

                return f"{prefix}.{rest}", False

            # resizer -> text_projection (the MLP resizer)
            if lk.startswith("resizer."):
                rest = lk[len("resizer.") :]
                return f"detector_model.text_projection.{rest}", False

            # Fallback for language backbone
            return f"detector_model.text_encoder.{lk}", False

        # --- DETR Transformer encoder ---
        if key.startswith("transformer.encoder.layers."):
            m3 = re.match(r"transformer\.encoder\.layers\.(\d+)\.(.+)", key)
            if m3:
                layer_idx = m3.group(1)
                rest = m3.group(2)
                prefix = f"detector_model.detr_encoder.layers.{layer_idx}"

                # self_attn.in_proj -> QKV split
                if "self_attn.in_proj_" in rest:
                    return None, False

                # cross_attn_image.in_proj -> QKV split
                if "cross_attn_image.in_proj_" in rest:
                    return None, False

                # cross_attn_image -> cross_attn
                rest = rest.replace("cross_attn_image", "cross_attn")

                # norm1/2/3 -> layer_norm1/2/3
                rest = re.sub(r"^norm(\d)", r"layer_norm\1", rest)

                return f"{prefix}.{rest}", False

        # encoder.norm -> detr_encoder output norm
        if key.startswith("transformer.encoder.norm."):
            rest = key[len("transformer.encoder.norm.") :]
            return f"detector_model.detr_encoder.layer_norm.{rest}", False

        # --- DETR Transformer decoder ---
        if key.startswith("transformer.decoder."):
            dk = key[len("transformer.decoder.") :]

            # decoder.layers.{N}
            m4 = re.match(r"layers\.(\d+)\.(.+)", dk)
            if m4:
                layer_idx = m4.group(1)
                rest = m4.group(2)
                prefix = f"detector_model.detr_decoder.layers.{layer_idx}"

                # self_attn.in_proj -> QKV split
                if "self_attn.in_proj_" in rest:
                    return None, False

                # cross_attn.in_proj -> QKV split
                if "cross_attn.in_proj_" in rest:
                    return None, False

                # ca_text.in_proj -> QKV split
                if "ca_text.in_proj_" in rest:
                    return None, False

                # ca_text -> text_cross_attn
                rest = rest.replace("ca_text", "text_cross_attn")
                # catext_norm -> text_cross_attn_layer_norm
                rest = rest.replace("catext_norm", "text_cross_attn_layer_norm")
                # cross_attn -> vision_cross_attn (but not cross_attn_layer_norm)
                # Be careful: cross_attn.out_proj -> vision_cross_attn.out_proj
                rest = re.sub(r"^cross_attn\b", "vision_cross_attn", rest)

                # norm1 -> self_attn_layer_norm
                rest = re.sub(r"^norm1\b", "self_attn_layer_norm", rest)
                # norm2 -> vision_cross_attn_layer_norm
                rest = re.sub(r"^norm2\b", "vision_cross_attn_layer_norm", rest)
                # norm3 -> mlp_layer_norm
                rest = re.sub(r"^norm3\b", "mlp_layer_norm", rest)

                return f"{prefix}.{rest}", False

            # bbox_embed.layers.{N} -> box_head.layer{N+1}
            m5 = re.match(r"bbox_embed\.layers\.(\d+)\.(.+)", dk)
            if m5:
                layer_idx = int(m5.group(1)) + 1
                rest = m5.group(2)
                return (
                    f"detector_model.detr_decoder.box_head.layer{layer_idx}.{rest}",
                    False,
                )

            # boxRPB_embed_x.layers.{N} -> box_rpb_embed_x.layer{N+1}
            m5b = re.match(r"boxRPB_embed_x\.layers\.(\d+)\.(.+)", dk)
            if m5b:
                layer_idx = int(m5b.group(1)) + 1
                rest = m5b.group(2)
                return (
                    f"detector_model.detr_decoder.box_rpb_embed_x.layer{layer_idx}.{rest}",
                    False,
                )

            # boxRPB_embed_y.layers.{N} -> box_rpb_embed_y.layer{N+1}
            m5c = re.match(r"boxRPB_embed_y\.layers\.(\d+)\.(.+)", dk)
            if m5c:
                layer_idx = int(m5c.group(1)) + 1
                rest = m5c.group(2)
                return (
                    f"detector_model.detr_decoder.box_rpb_embed_y.layer{layer_idx}.{rest}",
                    False,
                )

            # presence_token.weight
            if dk.startswith("presence_token."):
                rest = dk[len("presence_token.") :]
                return f"detector_model.detr_decoder.presence_token.{rest}", False

            # presence_token_head.layers.{N} -> presence_head.layer{N+1}
            m6 = re.match(r"presence_token_head\.layers\.(\d+)\.(.+)", dk)
            if m6:
                layer_idx = int(m6.group(1)) + 1
                rest = m6.group(2)
                return (
                    f"detector_model.detr_decoder.presence_head.layer{layer_idx}.{rest}",
                    False,
                )

            # presence_token_out_norm -> presence_layer_norm
            if dk.startswith("presence_token_out_norm."):
                rest = dk[len("presence_token_out_norm.") :]
                return f"detector_model.detr_decoder.presence_layer_norm.{rest}", False

            # query_embed
            if dk.startswith("query_embed."):
                rest = dk[len("query_embed.") :]
                return f"detector_model.detr_decoder.query_embed.{rest}", False

            # reference_points
            if dk.startswith("reference_points."):
                rest = dk[len("reference_points.") :]
                return f"detector_model.detr_decoder.reference_points.{rest}", False

            # ref_point_head.layers.{N} -> ref_point_head.layer{N+1}
            m7 = re.match(r"ref_point_head\.layers\.(\d+)\.(.+)", dk)
            if m7:
                layer_idx = int(m7.group(1)) + 1
                rest = m7.group(2)
                return (
                    f"detector_model.detr_decoder.ref_point_head.layer{layer_idx}.{rest}",
                    False,
                )

            # norm -> output_layer_norm
            if dk.startswith("norm."):
                rest = dk[len("norm.") :]
                return f"detector_model.detr_decoder.output_layer_norm.{rest}", False

            # Fallback decoder keys
            return f"detector_model.detr_decoder.{dk}", False

        # --- Dot product scoring ---
        if key.startswith("dot_prod_scoring."):
            dk = key[len("dot_prod_scoring.") :]
            # prompt_mlp.out_norm -> text_mlp_out_norm (must be before prompt_mlp)
            dk = dk.replace("prompt_mlp.out_norm", "text_mlp_out_norm")
            dk = dk.replace("prompt_mlp", "text_mlp")
            dk = dk.replace("prompt_proj", "text_proj")
            dk = dk.replace("hs_proj", "query_proj")
            return f"detector_model.dot_product_scoring.{dk}", False

        # --- Geometry encoder ---
        if key.startswith("geometry_encoder."):
            gk = key[len("geometry_encoder.") :]

            # QKV in geometry encoder
            if "self_attn.in_proj_" in gk or "cross_attn_image.in_proj_" in gk:
                return None, False

            # cross_attn_image -> cross_attn
            gk = gk.replace("cross_attn_image", "cross_attn")

            # norm1/2/3 -> layer_norm1/2/3
            gk = re.sub(r"\.norm(\d)\.", r".layer_norm\1.", gk)

            return f"detector_model.geometry_encoder.{gk}", False

        # --- Segmentation head ---
        if key.startswith("segmentation_head."):
            sk = key[len("segmentation_head.") :]

            # cross_attend_prompt.in_proj -> QKV split
            if "cross_attend_prompt.in_proj_" in sk:
                return None, False

            # pixel_decoder
            if sk.startswith("pixel_decoder."):
                rest = sk[len("pixel_decoder.") :]
                return f"detector_model.mask_decoder.pixel_decoder.{rest}", False

            # mask_predictor.mask_embed -> mask_embedder
            if sk.startswith("mask_predictor.mask_embed."):
                rest = sk[len("mask_predictor.mask_embed.") :]
                return f"detector_model.mask_decoder.mask_embedder.{rest}", False

            # cross_attend_prompt -> prompt_cross_attn
            if sk.startswith("cross_attend_prompt."):
                rest = sk[len("cross_attend_prompt.") :]
                return f"detector_model.mask_decoder.prompt_cross_attn.{rest}", False

            # cross_attn_norm -> prompt_cross_attn_norm
            if sk.startswith("cross_attn_norm."):
                rest = sk[len("cross_attn_norm.") :]
                return (
                    f"detector_model.mask_decoder.prompt_cross_attn_norm.{rest}",
                    False,
                )

            # instance_seg_head -> instance_projection
            if sk.startswith("instance_seg_head."):
                rest = sk[len("instance_seg_head.") :]
                return f"detector_model.mask_decoder.instance_projection.{rest}", False

            # semantic_seg_head -> semantic_projection
            if sk.startswith("semantic_seg_head."):
                rest = sk[len("semantic_seg_head.") :]
                return f"detector_model.mask_decoder.semantic_projection.{rest}", False

            # Fallback
            return f"detector_model.mask_decoder.{sk}", False

        # --- Fallback for any remaining detector keys ---
        return f"detector_model.{key}", False

    # ==================================================================
    # TRACKER keys
    # ==================================================================
    if key.startswith("tracker.model."):
        tk = key[len("tracker.model.") :]

        # --- Memory encoder (maskmem_backbone) ---
        if tk.startswith("maskmem_backbone."):
            mk = tk[len("maskmem_backbone.") :]

            # mask_downsampler.encoder.{N} -> mask_downsampler.layers.{N//3}.conv or .layer_norm
            # Last conv (layer_idx 4) maps to final_conv instead of layers.4
            m8 = re.match(r"mask_downsampler\.encoder\.(\d+)\.(.+)", mk)
            if m8:
                enc_idx = int(m8.group(1))
                rest = m8.group(2)
                layer_idx = enc_idx // 3
                sub_idx = enc_idx % 3
                if sub_idx == 0:
                    # Conv layer — layer 4 is the final 1x1 conv
                    if layer_idx == 4:
                        return (
                            f"tracker_model.memory_encoder.mask_downsampler.final_conv.{rest}",
                            False,
                        )
                    return (
                        f"tracker_model.memory_encoder.mask_downsampler.layers.{layer_idx}.conv.{rest}",
                        False,
                    )
                elif sub_idx == 1:
                    # LayerNorm
                    return (
                        f"tracker_model.memory_encoder.mask_downsampler.layers.{layer_idx}.layer_norm.{rest}",
                        False,
                    )
                else:
                    # Activation (sub_idx == 2), typically no params, skip
                    return None, True

            # fuser.layers.{N}
            m9 = re.match(r"fuser\.layers\.(\d+)\.(.+)", mk)
            if m9:
                layer_idx = m9.group(1)
                rest = m9.group(2)
                prefix = f"tracker_model.memory_encoder.memory_fuser.layers.{layer_idx}"

                rest = rest.replace("dwconv", "depthwise_conv")
                rest = rest.replace("gamma", "scale")
                rest = rest.replace("pwconv1", "pointwise_conv1")
                rest = rest.replace("pwconv2", "pointwise_conv2")
                # CXBlock uses layer_norm in MLX model
                if rest.startswith("norm."):
                    rest = "layer_norm." + rest[5:]

                return f"{prefix}.{rest}", False

            # pix_feat_proj -> feature_projection
            if mk.startswith("pix_feat_proj."):
                rest = mk[len("pix_feat_proj.") :]
                return f"tracker_model.memory_encoder.feature_projection.{rest}", False

            # Fallback memory encoder
            return f"tracker_model.memory_encoder.{mk}", False

        # --- Memory temporal positional encoding ---
        if tk.startswith("maskmem_tpos_enc"):
            rest = tk[len("maskmem_tpos_enc") :]
            return f"tracker_model.memory_temporal_positional_encoding{rest}", False

        # --- obj_ptr_tpos_proj ---
        if tk.startswith("obj_ptr_tpos_proj."):
            rest = tk[len("obj_ptr_tpos_proj.") :]
            return (
                f"tracker_model.temporal_positional_encoding_projection_layer.{rest}",
                False,
            )

        # --- Memory attention (transformer.encoder) ---
        if tk.startswith("transformer.encoder.layers."):
            m10 = re.match(r"transformer\.encoder\.layers\.(\d+)\.(.+)", tk)
            if m10:
                layer_idx = m10.group(1)
                rest = m10.group(2)
                prefix = f"tracker_model.memory_attention.layers.{layer_idx}"

                # QKV in memory attention
                if "self_attn.in_proj_" in rest or "cross_attention.in_proj_" in rest:
                    return None, False

                return f"{prefix}.{rest}", False

        # transformer.encoder.norm -> memory_attention.layer_norm
        if tk.startswith("transformer.encoder.norm."):
            rest = tk[len("transformer.encoder.norm.") :]
            return f"tracker_model.memory_attention.layer_norm.{rest}", False

        # --- Interactive SAM components (keep paths mostly intact) ---
        # interactive_sam_mask_decoder, interactive_sam_prompt_encoder, etc.
        if tk.startswith("interactive_sam_mask_decoder."):
            rest = tk[len("interactive_sam_mask_decoder.") :]
            # QKV in SAM mask decoder
            if ".in_proj_" in rest:
                return None, False
            return f"tracker_model.interactive_sam_mask_decoder.{rest}", False

        if tk.startswith("interactive_sam_prompt_encoder."):
            rest = tk[len("interactive_sam_prompt_encoder.") :]
            # Remap mask_embed sequential indices to named convs
            mask_embed_map = {
                "mask_embed.0.": "mask_embed.conv1.",
                "mask_embed.1.": "mask_embed.layer_norm1.",
                "mask_embed.3.": "mask_embed.conv2.",
                "mask_embed.4.": "mask_embed.layer_norm2.",
                "mask_embed.6.": "mask_embed.conv3.",
            }
            for old, new in mask_embed_map.items():
                if rest.startswith(old):
                    rest = rest.replace(old, new, 1)
                    break
            return f"tracker_model.interactive_sam_prompt_encoder.{rest}", False

        if tk.startswith("interactive_obj_ptr_proj."):
            rest = tk[len("interactive_obj_ptr_proj.") :]
            return f"tracker_model.interactive_obj_ptr_proj.{rest}", False

        # --- SAM mask decoder (non-interactive) ---
        if tk.startswith("sam_mask_decoder."):
            rest = tk[len("sam_mask_decoder.") :]
            if ".in_proj_" in rest:
                return None, False
            return f"tracker_model.sam_mask_decoder.{rest}", False

        if tk.startswith("sam_prompt_encoder."):
            rest = tk[len("sam_prompt_encoder.") :]
            return f"tracker_model.sam_prompt_encoder.{rest}", False

        # obj_ptr_proj
        if tk.startswith("obj_ptr_proj."):
            rest = tk[len("obj_ptr_proj.") :]
            return f"tracker_model.obj_ptr_proj.{rest}", False

        # memory_encoder
        if tk.startswith("memory_encoder."):
            rest = tk[len("memory_encoder.") :]
            return f"tracker_model.memory_encoder.{rest}", False

        # --- Image encoder (shared with detector potentially) ---
        if tk.startswith("image_encoder."):
            rest = tk[len("image_encoder.") :]
            return f"tracker_model.image_encoder.{rest}", False

        # --- Fallback tracker keys ---
        return f"tracker_model.{tk}", False

    # ==================================================================
    # Keys not matching detector or tracker -- pass through
    # ==================================================================
    return key, False


def _get_qkv_base_and_suffix(old_key):
    """Extract the base path and weight/bias suffix from a QKV key.

    For 'foo.attn.qkv.weight' returns ('foo.attn', 'qkv', 'weight')
    For 'foo.attn.in_proj_weight' returns ('foo.attn', 'in_proj', 'weight')
    """
    if ".qkv." in old_key:
        m = re.match(r"(.+)\.qkv\.(weight|bias)$", old_key)
        if m:
            return m.group(1), "qkv", m.group(2)
    if ".in_proj_" in old_key:
        m = re.match(r"(.+)\.in_proj_(weight|bias)$", old_key)
        if m:
            return m.group(1), "in_proj", m.group(2)
    return None, None, None


def _resolve_qkv_prefix(old_key):
    """Directly resolve the new MLX prefix for a fused QKV key.

    Given a full old key like:
      detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight
    Returns the new prefix like:
      detector_model.vision_encoder.backbone.layers.0.attention
    """
    key = old_key

    # ViT attention: detector.backbone.vision_backbone.trunk.blocks.{N}.attn.qkv.*
    m = re.match(
        r"detector\.backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn\.qkv\.",
        key,
    )
    if m:
        return f"detector_model.vision_encoder.backbone.layers.{m.group(1)}.attention"

    # CLIP text encoder: detector.backbone.language_backbone.encoder.transformer.resblocks.{N}.attn.in_proj_*
    m = re.match(
        r"detector\.backbone\.language_backbone\.encoder\.transformer\.resblocks\.(\d+)\.attn\.in_proj_",
        key,
    )
    if m:
        return f"detector_model.text_encoder.text_model.encoder.layers.{m.group(1)}.self_attn"

    # DETR encoder self_attn: detector.transformer.encoder.layers.{N}.self_attn.in_proj_*
    m = re.match(
        r"detector\.transformer\.encoder\.layers\.(\d+)\.self_attn\.in_proj_",
        key,
    )
    if m:
        return f"detector_model.detr_encoder.layers.{m.group(1)}.self_attn"

    # DETR encoder cross_attn_image: detector.transformer.encoder.layers.{N}.cross_attn_image.in_proj_*
    m = re.match(
        r"detector\.transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.in_proj_",
        key,
    )
    if m:
        return f"detector_model.detr_encoder.layers.{m.group(1)}.cross_attn"

    # DETR decoder self_attn: detector.transformer.decoder.layers.{N}.self_attn.in_proj_*
    m = re.match(
        r"detector\.transformer\.decoder\.layers\.(\d+)\.self_attn\.in_proj_",
        key,
    )
    if m:
        return f"detector_model.detr_decoder.layers.{m.group(1)}.self_attn"

    # DETR decoder cross_attn: detector.transformer.decoder.layers.{N}.cross_attn.in_proj_*
    m = re.match(
        r"detector\.transformer\.decoder\.layers\.(\d+)\.cross_attn\.in_proj_",
        key,
    )
    if m:
        return f"detector_model.detr_decoder.layers.{m.group(1)}.vision_cross_attn"

    # DETR decoder ca_text: detector.transformer.decoder.layers.{N}.ca_text.in_proj_*
    m = re.match(
        r"detector\.transformer\.decoder\.layers\.(\d+)\.ca_text\.in_proj_",
        key,
    )
    if m:
        return f"detector_model.detr_decoder.layers.{m.group(1)}.text_cross_attn"

    # Geometry encoder self_attn: detector.geometry_encoder.*.self_attn.in_proj_*
    m = re.match(
        r"detector\.geometry_encoder\.(.+?)\.self_attn\.in_proj_",
        key,
    )
    if m:
        rest = m.group(1)
        return f"detector_model.geometry_encoder.{rest}.self_attn"

    # Geometry encoder cross_attn_image: detector.geometry_encoder.*.cross_attn_image.in_proj_*
    m = re.match(
        r"detector\.geometry_encoder\.(.+?)\.cross_attn_image\.in_proj_",
        key,
    )
    if m:
        rest = m.group(1)
        return f"detector_model.geometry_encoder.{rest}.cross_attn"

    # Segmentation head cross_attend_prompt: detector.segmentation_head.cross_attend_prompt.in_proj_*
    m = re.match(
        r"detector\.segmentation_head\.cross_attend_prompt\.in_proj_",
        key,
    )
    if m:
        return "detector_model.mask_decoder.prompt_cross_attn"

    # Tracker memory attention self_attn: tracker.model.transformer.encoder.layers.{N}.self_attn.in_proj_*
    m = re.match(
        r"tracker\.model\.transformer\.encoder\.layers\.(\d+)\.self_attn\.in_proj_",
        key,
    )
    if m:
        return f"tracker_model.memory_attention.layers.{m.group(1)}.self_attn"

    # Tracker memory attention cross_attention: tracker.model.transformer.encoder.layers.{N}.cross_attention.in_proj_*
    m = re.match(
        r"tracker\.model\.transformer\.encoder\.layers\.(\d+)\.cross_attention\.in_proj_",
        key,
    )
    if m:
        return f"tracker_model.memory_attention.layers.{m.group(1)}.cross_attention"

    # Tracker interactive SAM mask decoder: tracker.model.interactive_sam_mask_decoder.*.in_proj_*
    m = re.match(
        r"tracker\.model\.interactive_sam_mask_decoder\.(.+?)\.in_proj_",
        key,
    )
    if m:
        rest = m.group(1)
        return f"tracker_model.interactive_sam_mask_decoder.{rest}"

    # Tracker SAM mask decoder: tracker.model.sam_mask_decoder.*.in_proj_*
    m = re.match(
        r"tracker\.model\.sam_mask_decoder\.(.+?)\.in_proj_",
        key,
    )
    if m:
        rest = m.group(1)
        return f"tracker_model.sam_mask_decoder.{rest}"

    return None


def convert_weights(checkpoint):
    """Convert all weights from Meta format to MLX format."""
    converted = {}

    # First pass: collect QKV pairs (weight + bias for same key base)
    qkv_groups = {}  # base_old_key -> {'weight': tensor, 'bias': tensor}

    keys_to_process = []
    for old_key in checkpoint:
        if "freqs_cis" in old_key:
            continue

        base, qkv_type, suffix = _get_qkv_base_and_suffix(old_key)
        if base is not None and qkv_type is not None:
            # This is a QKV key
            # Build the group key from the full old key minus the suffix
            if qkv_type == "qkv":
                group_key = old_key.rsplit(".qkv.", 1)[0]
            else:
                group_key = old_key.rsplit(".in_proj_", 1)[0]

            if group_key not in qkv_groups:
                qkv_groups[group_key] = {"old_key_sample": old_key}
            qkv_groups[group_key][suffix] = checkpoint[old_key]
        else:
            keys_to_process.append(old_key)

    # Process QKV groups
    for group_key, group_data in qkv_groups.items():
        old_key_sample = group_data["old_key_sample"]
        weight = group_data.get("weight")
        bias = group_data.get("bias")

        if weight is None:
            continue

        weights_split, biases_split = split_qkv(
            weight.numpy(), bias.numpy() if bias is not None else None
        )

        # Determine the new prefix using direct resolution
        new_prefix = _resolve_qkv_prefix(old_key_sample)

        if new_prefix is None:
            print(f"WARNING: Could not map QKV key: {old_key_sample}")
            continue

        for proj_name, w in zip(["q_proj", "k_proj", "v_proj"], weights_split):
            converted[f"{new_prefix}.{proj_name}.weight"] = w

        if biases_split is not None:
            for proj_name, b in zip(["q_proj", "k_proj", "v_proj"], biases_split):
                converted[f"{new_prefix}.{proj_name}.bias"] = b

    # Process non-QKV keys
    for old_key in keys_to_process:
        new_key, skip = convert_key(old_key)
        if skip:
            continue
        if new_key is None:
            # Should not happen for non-QKV keys, but just in case
            print(f"WARNING: Unmapped key: {old_key}")
            continue

        tensor = checkpoint[old_key]
        value = tensor.numpy()

        # Conv2d / ConvTranspose2d transpose
        # Skip 4D tensors that are not conv weights (e.g., positional encodings)
        skip_4d_transpose = (
            "positional_encoding" in new_key
            or "tpos_enc" in old_key
            or "pos_embed" in old_key
        )
        if value.ndim == 4 and not skip_4d_transpose:
            if is_conv_transpose(old_key):
                # ConvTranspose2d: (in, out, H, W) -> (out, H, W, in)
                value = value.transpose(1, 2, 3, 0)
            else:
                # Conv2d: (out, in, H, W) -> (out, H, W, in)
                value = value.transpose(0, 2, 3, 1)

        # CLS token strip from pos_embed
        if (
            "position_embeddings" in new_key
            and value.ndim >= 2
            and value.shape[-2] == 577
        ):
            # Strip CLS token at index 0: (1, 577, 1024) -> (1, 576, 1024)
            value = value[..., 1:, :]

        # Text projection transpose
        if new_key == "detector_model.text_encoder.text_projection.weight":
            value = value.T

        converted[new_key] = value

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    # Concat tracker point embeddings: point_embeddings.{0,1,2,3} -> point_embed
    for prompt_enc_prefix in [
        "tracker_model.interactive_sam_prompt_encoder",
        "tracker_model.sam_prompt_encoder",
    ]:
        point_weights = []
        keys_to_remove = []
        for i in range(4):
            pe_key = f"{prompt_enc_prefix}.point_embeddings.{i}.weight"
            if pe_key in converted:
                point_weights.append(converted[pe_key])
                keys_to_remove.append(pe_key)

        if len(point_weights) == 4:
            concat_key = f"{prompt_enc_prefix}.point_embed.weight"
            converted[concat_key] = np.concatenate(point_weights, axis=0)
            for k in keys_to_remove:
                del converted[k]

    # Copy shared_image_embedding from interactive_sam_prompt_encoder
    pe_gauss_key = "tracker_model.interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    if pe_gauss_key in converted:
        converted["tracker_model.shared_image_embedding.positional_embedding"] = (
            converted[pe_gauss_key].copy()
        )

    return converted


def save_config(output_path):
    """Download and save config.json from facebook/sam3.1."""
    try:
        config_path = hf_hub_download("facebook/sam3.1", "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        # Fix model_type for mlx-vlm routing
        config["model_type"] = "sam3.1_video"
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config.json from facebook/sam3.1")
    except Exception as e:
        print(f"Could not download config.json: {e}")
        print("You may need to manually provide a config.json")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM 3.1 Meta checkpoint to MLX safetensors format."
    )
    parser.add_argument("--output", default="/tmp/sam3.1-mlx", help="Output directory")
    args = parser.parse_args()

    print("Loading checkpoint...")
    checkpoint = load_checkpoint()
    print(f"Loaded {len(checkpoint)} keys")

    print("Converting weights...")
    converted = convert_weights(checkpoint)
    print(f"Converted to {len(converted)} keys")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Saving safetensors...")
    save_file(converted, str(output_path / "model.safetensors"))

    save_config(output_path)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()
