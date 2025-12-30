from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Dict, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


# -----------------------------------------------------------------------------
# Robust converters (torch / numpy / list -> mx.array)
# -----------------------------------------------------------------------------
def _as_np(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, mx.array):
        return np.array(x)

    # torch.Tensor -> numpy
    try:
        import torch  # optional
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    # generic ".numpy()"
    if hasattr(x, "numpy") and callable(x.numpy):
        try:
            return x.numpy()
        except Exception:
            pass

    # list/tuple/scalar
    try:
        return np.asarray(x)
    except Exception:
        return None


def _as_mx(x: Any) -> Any:
    """
    Convert to mx.array when possible; otherwise return as-is (e.g. None).
    """
    if x is None:
        return None
    if isinstance(x, mx.array):
        return x
    arr = _as_np(x)
    if arr is None:
        return x
    return mx.array(arr)


def _as_mx_int32(x: Any) -> Optional[mx.array]:
    if x is None:
        return None
    if isinstance(x, mx.array):
        return x if x.dtype == mx.int32 else x.astype(mx.int32)
    arr = _as_np(x)
    if arr is None:
        raise ValueError("Failed to convert to numpy for int32 conversion.")
    return mx.array(arr.astype(np.int32))


def _as_mx_bool(x: Any) -> Optional[mx.array]:
    if x is None:
        return None
    if isinstance(x, mx.array):
        return x if x.dtype == mx.bool_ else x.astype(mx.bool_)
    arr = _as_np(x)
    if arr is None:
        raise ValueError("Failed to convert to numpy for bool conversion.")
    return mx.array(arr.astype(np.bool_))


# -----------------------------------------------------------------------------
# Small utils
# -----------------------------------------------------------------------------
def l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    denom = mx.sqrt(mx.maximum((x * x).sum(axis=-1, keepdims=True), eps))
    return x / denom


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
) -> mx.array:
    """
    Scatter image features into final_embedding where mask is True.
    Compatible with MLX versions that do NOT support .at[].set().
    """
    final_shape = final_embedding.shape

    img_flat = mx.flatten(scaled_image_features)
    out_flat = mx.flatten(final_embedding)
    mask_flat = mx.flatten(image_mask_expanded)

    pos_np = np.where(np.array(mask_flat))[0].astype(np.uint32)
    pos = mx.array(pos_np, dtype=mx.uint32)

    #  MLX-compatible assignment (no .at[].set())
    out_flat[pos] = img_flat

    return mx.reshape(out_flat, final_shape)


# -----------------------------------------------------------------------------
# Backbone (Qwen3-VL style) for multimodal mixing + logits
# -----------------------------------------------------------------------------
class VLMBackbone(nn.Module):
    """
    Qwen3-VL backbone used by mlx-vlm:
      - embeds tokens
      - injects image features into <image>/<video> token positions
      - runs LanguageModel to produce logits (for generation)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Any,
        pixel_values: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        """
        Returns:
          inputs_embeds: [B, T, H]
          visual_pos_masks: [B, T] bool mask where visual tokens are
          deepstack_visual_embeds: list[tensor] for deepstack (may be None)
        """
        input_ids_mx = _as_mx_int32(input_ids)

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if grid_thw is not None:
            grid_thw = _as_mx_int32(grid_thw)

        # text-only
        if pixel_values is None:
            return {
                "inputs_embeds": self.language_model.model.embed_tokens(input_ids_mx),
                "visual_pos_masks": None,
                "deepstack_visual_embeds": None,
            }

        # ensure mx.array (torch tensor fix)
        pixel_values_mx = _as_mx(pixel_values)

        # cast to vision proj dtype
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values_mx = pixel_values_mx.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids_mx)
        image_features, deepstack_image_embeds = self.vision_tower(pixel_values_mx, grid_thw)

        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids_mx,
            image_token_index=self.config.image_token_index,
            video_token_index=self.config.video_token_index,
        )

        visual_pos_masks = image_mask[..., 0].astype(mx.bool_)

        return {
            "inputs_embeds": inputs_embeds,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_image_embeds,
        }

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        image_token_index: int,
        video_token_index: int,
    ):
        special_mask = (input_ids == image_token_index) | (input_ids == video_token_index)

        special_mask = special_mask[..., None]
        special_mask = mx.broadcast_to(special_mask, inputs_embeds.shape)

        n_mask_elements = int(special_mask.sum().item()) if hasattr(special_mask.sum(), "item") else int(special_mask.sum())
        if n_mask_elements != int(image_features.size):
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"mask_elems={n_mask_elements}, image_features.size={int(image_features.size)}"
            )

        out = masked_scatter(inputs_embeds, special_mask, image_features)
        return out, special_mask

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: Any,
        pixel_values: Optional[Any] = None,
        mask: Optional[Any] = None,
        cache=None,
        **kwargs,
    ):
        pack = self.get_input_embeddings(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        kwargs.update({"pixel_values": _as_mx(pixel_values), **pack})
        logits = self.language_model(_as_mx_int32(input_ids), mask=_as_mx_int32(mask) if mask is not None else None, cache=cache, **kwargs)
        return logits


# -----------------------------------------------------------------------------
# ColQwen3 wrapper (Tomoro) - token-level embeddings + MaxSim
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    ColQwen3 MLX wrapper:
      - keeps VLM backbone (logits path)
      - adds embedding_proj_layer to output token embeddings [B,T,D]
      - provides encode / encode_queries / encode_images and maxsim

    IMPORTANT:
      - deepstack injection is disabled by default here (use_deepstack=False),
        because current deepstack wiring can mismatch shapes and crash on MLX/Metal.
        Once language.py deepstack indexing is fixed, you can enable it.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # backbone behaves like qwen3_vl internally
        backbone_cfg = replace(config, model_type="qwen3_vl")
        self.vlm = VLMBackbone(backbone_cfg)

        hidden = config.text_config.hidden_size
        self.embedding_proj_layer = nn.Linear(hidden, config.embed_dim, bias=True)

    # generation/logits path
    def __call__(
        self,
        input_ids: Any,
        pixel_values: Optional[Any] = None,
        mask: Optional[Any] = None,
        cache=None,
        **kwargs,
    ):
        return self.vlm(_as_mx_int32(input_ids), pixel_values=pixel_values, mask=mask, cache=cache, **kwargs)

    # -------------------------------------------------------------------------
    # HF -> MLX weight key mapping
    # -------------------------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            # HF: vlm.model.language_model.* -> MLX: vlm.language_model.model.*
            if k.startswith("vlm.model.language_model."):
                nk = "vlm.language_model.model." + k[len("vlm.model.language_model.") :]
                out[nk] = v
                continue

            # HF: vlm.lm_head.* -> MLX: vlm.language_model.lm_head.*
            if k.startswith("vlm.lm_head."):
                nk = "vlm.language_model.lm_head." + k[len("vlm.lm_head.") :]
                out[nk] = v
                continue

            # HF: vlm.model.visual.* -> MLX: vlm.vision_tower.*
            if k.startswith("vlm.model.visual."):
                nk = "vlm.vision_tower." + k[len("vlm.model.visual.") :]
                out[nk] = v
                continue

            # HF: vlm.model.vision_tower.* -> MLX: vlm.vision_tower.*
            if k.startswith("vlm.model.vision_tower."):
                nk = "vlm.vision_tower." + k[len("vlm.model.vision_tower.") :]
                out[nk] = v
                continue

            # projection layer
            if k.startswith("embedding_proj_layer."):
                out[k] = v
                continue

            # fallback old names (just in case)
            if k.startswith("model.language_model."):
                nk = "vlm.language_model.model." + k[len("model.language_model.") :]
                out[nk] = v
                continue
            if k.startswith("model.visual."):
                nk = "vlm.vision_tower." + k[len("model.visual.") :]
                out[nk] = v
                continue

            out[k] = v

        return out

    # -------------------------------------------------------------------------
    # Embedding forward
    # -------------------------------------------------------------------------
    def encode(
        self,
        input_ids: Any,
        pixel_values: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        only_visual: bool = False,
        use_deepstack: bool = False,
        **kwargs,
    ) -> dict:
        """
        Returns token-level embeddings (ColBERT-style).

        Output:
          embeddings: [B,T,D] padded
          embeddings_list: list length B, each [Ti,D] trimmed
          attention_mask: mx.int32 if provided
          visual_mask: [B,T] bool if multimodal
        """
        input_ids_mx = _as_mx_int32(input_ids)
        attn_mx = _as_mx_int32(attention_mask) if attention_mask is not None else None
        pixel_mx = _as_mx(pixel_values) if pixel_values is not None else None

        # grids if present
        if "image_grid_thw" in kwargs and kwargs["image_grid_thw"] is not None:
            kwargs["image_grid_thw"] = _as_mx_int32(kwargs["image_grid_thw"])
        if "video_grid_thw" in kwargs and kwargs["video_grid_thw"] is not None:
            kwargs["video_grid_thw"] = _as_mx_int32(kwargs["video_grid_thw"])

        pack = self.vlm.get_input_embeddings(input_ids=input_ids_mx, pixel_values=pixel_mx, **kwargs)

        deepstack = pack.get("deepstack_visual_embeds", None) if use_deepstack else None
        visual_mask = pack.get("visual_pos_masks", None)

        h = self.vlm.language_model.forward_hidden(
            input_ids_mx,
            inputs_embeds=pack["inputs_embeds"],
            mask=attn_mx,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            pixel_values=pixel_mx,
            **kwargs,
        )

        e = l2_normalize(self.embedding_proj_layer(h))
        mx.eval(e)  # materialize to reduce lazy graph buildup / Metal crashes

        # build per-sample lists
        embs_list: List[mx.array] = []

        if only_visual and (visual_mask is not None):
            for b in range(int(e.shape[0])):
                idx_np = np.where(np.array(visual_mask[b]))[0].astype(np.uint32)
                idx_mx = mx.array(idx_np, dtype=mx.uint32)
                embs_list.append(e[b][idx_mx])
            return {
                "embeddings": e,
                "embeddings_list": embs_list,
                "attention_mask": attn_mx,
                "visual_mask": visual_mask,
            }

        if attn_mx is not None:
            for b in range(int(e.shape[0])):
                idx_np = np.where(np.array(attn_mx[b]))[0].astype(np.uint32)
                idx_mx = mx.array(idx_np, dtype=mx.uint32)
                embs_list.append(e[b][idx_mx])
        else:
            for b in range(int(e.shape[0])):
                embs_list.append(e[b])

        return {
            "embeddings": e,
            "embeddings_list": embs_list,
            "attention_mask": attn_mx,
            "visual_mask": visual_mask,
        }

    # -------------------------------------------------------------------------
    # MaxSim scoring
    # -------------------------------------------------------------------------
    @staticmethod
    def maxsim(q_emb: mx.array, d_emb: mx.array, chunk: int = 1024) -> mx.array:
        """
        ColBERT MaxSim (chunked + fp32 for stability):
          score(q,d) = sum_i max_j <q_i, d_j>
        """
        q = q_emb.astype(mx.float32)
        d = d_emb.astype(mx.float32)

        Tq = int(q.shape[0])
        Td = int(d.shape[0])

        token_max = mx.full((Tq,), -1e9, dtype=mx.float32)

        for j in range(0, Td, chunk):
            dj = d[j : j + chunk]            # [c, D]
            sim = q @ dj.T                   # [Tq, c]
            token_max = mx.maximum(token_max, sim.max(axis=1))

        s = token_max.sum()
        mx.eval(s)
        return s

    # -------------------------------------------------------------------------
    # Convenience helpers (processor integrated)
    # -------------------------------------------------------------------------
    def encode_queries(self, processor, texts: list[str], batch_size: int = 8, use_deepstack: bool = False):
        """
        Returns list of [T,D] per query (padding removed).
        """
        outs = []
        for s in range(0, len(texts), batch_size):
            batch = processor.process_texts(texts=texts[s : s + batch_size])

            input_ids = _as_mx_int32(batch["input_ids"])
            attn = _as_mx_int32(batch.get("attention_mask")) if "attention_mask" in batch else None

            out = self.encode(
                input_ids=input_ids,
                attention_mask=attn,
                only_visual=False,
                use_deepstack=use_deepstack,
            )
            outs.extend(out["embeddings_list"])
        return outs

    def encode_images(self, processor, images, batch_size: int = 2, use_deepstack: bool = False):
        """
        Returns list of [Tv,D] per image (visual tokens only).
        batch_size default=2 to reduce Metal memory pressure.
        """
        outs = []
        for s in range(0, len(images), batch_size):
            batch_imgs = images[s : s + batch_size]
            feats = processor.process_images(images=batch_imgs)

            input_ids = _as_mx_int32(feats["input_ids"])
            pixel_values = _as_mx(feats.get("pixel_values"))
            attn = _as_mx_int32(feats.get("attention_mask")) if "attention_mask" in feats else None

            kwargs = {}
            if "image_grid_thw" in feats and feats["image_grid_thw"] is not None:
                kwargs["image_grid_thw"] = _as_mx_int32(feats["image_grid_thw"])
            if "video_grid_thw" in feats and feats["video_grid_thw"] is not None:
                kwargs["video_grid_thw"] = _as_mx_int32(feats["video_grid_thw"])

            out = self.encode(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attn,
                only_visual=True,
                use_deepstack=use_deepstack,
                **kwargs,
            )
            outs.extend(out["embeddings_list"])
        return outs