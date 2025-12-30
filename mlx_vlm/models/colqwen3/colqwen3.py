from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


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
    Scatter image features into input embedding positions where image_mask_expanded is True.
    Works with flattened indexing to match how mlx-vlm does it.
    """
    final_shape = final_embedding.shape

    img_flat = mx.flatten(scaled_image_features)
    out_flat = mx.flatten(final_embedding)
    mask_flat = mx.flatten(image_mask_expanded)

    pos = mx.array(np.where(np.array(mask_flat))[0], mx.uint32)
    out_flat[pos] = img_flat
    return mx.reshape(out_flat, final_shape)


def _to_mx(x: Any):
    """
    Robust converter for processor outputs:
      - np.ndarray / list / tuple -> mx.array
      - already mx.array -> keep
      - others -> keep
    """
    if isinstance(x, mx.array):
        return x
    if isinstance(x, np.ndarray):
        return mx.array(x)
    if isinstance(x, (list, tuple)):
        return mx.array(x)
    return x


# -----------------------------------------------------------------------------
# Backbone (Qwen3-VL style) for multimodal tokenization / logits
# -----------------------------------------------------------------------------
class VLMBackbone(nn.Module):
    """
    This is the Qwen3-VL "logits" backbone as used by mlx-vlm.
    We keep it because:
      - it already supports visual token injection into text embeddings
      - it supports deepstack visual features used by Qwen3-VL
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> dict:
        """
        Returns:
          inputs_embeds: [B, T, H]
          visual_pos_masks: [B, T] bool mask where visual tokens are
          deepstack_visual_embeds: list of tensors for deepstack injection
        """
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        # text-only
        if pixel_values is None:
            return {
                "inputs_embeds": self.language_model.model.embed_tokens(input_ids),
                "visual_pos_masks": None,
                "deepstack_visual_embeds": None,
            }

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        image_features, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw)

        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            image_features=image_features,
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            image_token_index=self.config.image_token_index,
            video_token_index=self.config.video_token_index,
        )

        # image_mask is broadcasted to [B, T, H]; return [B, T]
        visual_pos_masks = image_mask[..., 0]

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
        n_image_tokens = special_mask.sum()

        special_mask = special_mask[..., None]
        special_mask = mx.broadcast_to(special_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_mask_elements = special_mask.sum()

        if n_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={n_image_features}"
            )

        out = masked_scatter(inputs_embeds, special_mask, image_features)
        return out, special_mask

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
        pack = self.get_input_embeddings(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
        kwargs.update({"pixel_values": pixel_values, **pack})
        logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
        return logits


# -----------------------------------------------------------------------------
# ColQwen3 Wrapper (Tomoro) - exposes multi-vector embeddings + MaxSim
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    ColQwen3 MLX model:
      - keeps a VLM backbone for multimodal mixing
      - adds embedding_proj_layer to produce token-level embeddings [B,T,D]
      - provides encode(), encode_queries(), encode_images(), maxsim()

    Notes:
      - We intentionally return token-level vectors (ColBERT style).
      - User can apply MaxSim downstream.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # backbone should behave like Qwen3-VL internally
        backbone_cfg = replace(config, model_type="qwen3_vl")
        self.vlm = VLMBackbone(backbone_cfg)

        hidden = config.text_config.hidden_size
        self.embedding_proj_layer = nn.Linear(hidden, config.embed_dim, bias=True)

    # -------------------------------------------------------------------------
    # Standard forward (logits) - used by mlx_vlm.generate
    # -------------------------------------------------------------------------
    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        return self.vlm(input_ids, pixel_values=pixel_values, mask=mask, cache=cache, **kwargs)

    # -------------------------------------------------------------------------
    # Weight-key sanitizer for HF -> MLX mapping
    # -------------------------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        """
        Map HF checkpoint keys to MLX module names.

        HF side may contain:
          - vlm.model.language_model.*  (or other nested names)
          - vlm.lm_head.*
          - vlm.model.visual.*
          - embedding_proj_layer.*
        """
        out = {}
        for k, v in weights.items():
            # HF: vlm.model.language_model.layers... -> MLX: vlm.language_model.model.layers...
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

            # projection layer (ColQwen3 specific)
            if k.startswith("embedding_proj_layer."):
                out[k] = v
                continue

            # extra historical mappings (just in case)
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
    # Embedding forward: token-level multi-vector
    # -------------------------------------------------------------------------
    def encode(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        only_visual: bool = False,
        **kwargs,
    ) -> dict:
        """
        Returns token-level multi-vector embeddings (ColBERT style).

        Output:
          embeddings: [B, T, D]  (padded)
          embeddings_list: list of length B, each is [Ti, D] trimmed
          attention_mask: original attention_mask (if given)
          visual_mask: [B, T] True where visual tokens are (if any)
        """
        pack = self.vlm.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        # hidden: [B, T, H]
        h = self.vlm.language_model.forward_hidden(
            input_ids,
            inputs_embeds=pack["inputs_embeds"],
            mask=attention_mask,  # for consistency; causal right-padding OK
            visual_pos_masks=pack.get("visual_pos_masks", None),
            deepstack_visual_embeds=pack.get("deepstack_visual_embeds", None),
            pixel_values=pixel_values,
            **kwargs,
        )

        # project -> [B, T, D]
        e = self.embedding_proj_layer(h)
        e = l2_normalize(e)

        visual_mask = pack.get("visual_pos_masks", None)

        # Build per-sample lists
        embs_list = []

        if only_visual and visual_mask is not None:
            for b in range(e.shape[0]):
                idx = mx.array(np.where(np.array(visual_mask[b]))[0], mx.uint32)
                embs_list.append(e[b][idx])
            return {
                "embeddings": e,
                "embeddings_list": embs_list,
                "attention_mask": attention_mask,
                "visual_mask": visual_mask,
            }

        # Default: trim by attention_mask if available (remove padding)
        if attention_mask is not None:
            for b in range(e.shape[0]):
                idx = mx.array(np.where(np.array(attention_mask[b]))[0], mx.uint32)
                embs_list.append(e[b][idx])
        else:
            for b in range(e.shape[0]):
                embs_list.append(e[b])

        return {
            "embeddings": e,
            "embeddings_list": embs_list,
            "attention_mask": attention_mask,
            "visual_mask": visual_mask,
        }

    # -------------------------------------------------------------------------
    # ColBERT MaxSim
    # -------------------------------------------------------------------------
    @staticmethod
    def maxsim(q_emb: mx.array, d_emb: mx.array) -> mx.array:
        """
        score(q,d) = sum_i max_j <q_i, d_j>
        q_emb: [Tq, D]
        d_emb: [Td, D]
        """
        sim = q_emb @ d_emb.T
        return sim.max(axis=1).sum()

    # -------------------------------------------------------------------------
    # High-level helpers (processor-integrated)
    # -------------------------------------------------------------------------
    def encode_queries(self, processor, texts: list[str], batch_size: int = 8):
        """
        Returns: list of token embeddings per query: each is [T, D] (trimmed).
        """
        outs = []
        for s in range(0, len(texts), batch_size):
            batch = processor.process_texts(texts=texts[s : s + batch_size])

            input_ids = _to_mx(batch["input_ids"])
            attn = _to_mx(batch.get("attention_mask")) if "attention_mask" in batch else None

            out = self.encode(input_ids=input_ids, attention_mask=attn)
            outs.extend(out["embeddings_list"])
        return outs

    def encode_images(self, processor, images, batch_size: int = 4):
        """
        Returns: list of visual-token embeddings per image: each is [Tv, D].

        Perfect for: PDF pages -> patch -> PIL -> embeddings
        """
        outs = []
        for s in range(0, len(images), batch_size):
            batch_imgs = images[s : s + batch_size]
            feats = processor.process_images(images=batch_imgs)

            input_ids = _to_mx(feats["input_ids"])
            pixel_values = _to_mx(feats.get("pixel_values"))
            attn = _to_mx(feats.get("attention_mask")) if "attention_mask" in feats else None

            kwargs = {}
            if "image_grid_thw" in feats:
                kwargs["image_grid_thw"] = _to_mx(feats["image_grid_thw"])
            if "video_grid_thw" in feats:
                kwargs["video_grid_thw"] = _to_mx(feats["video_grid_thw"])

            out = self.encode(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attn,
                only_visual=True,
                **kwargs,
            )
            outs.extend(out["embeddings_list"])
        return outs