from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .audio import AudioModel
from .config import ModelConfig
from .language import LanguageModel, fuse_qkvr, shared_experts_to_dense
from .vision import VisionModel


def masked_scatter(input_tensor, mask, source):
    """MLX port of torch.Tensor.masked_scatter: fill mask==True slots with source in order."""
    mask = mask.astype(mx.bool_)
    if not mask.any():
        return input_tensor
    shape = input_tensor.shape
    flat = input_tensor.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()
    positions = mx.cumsum(mask_flat.astype(mx.int32)) - 1
    positions = mx.clip(positions, 0, source_flat.size - 1)
    selected = source_flat[positions]
    return mx.where(mask_flat, selected, flat).reshape(shape)


def _split_gate_up(v):
    """De-interleave a fused [..., 2*inter, hidden] gate/up weight into (gate, up).

    Return contiguous halves for performance; `gather_mm / `gather_qmm` are much slower when fed strided views
    """
    *lead, two_i, hidden = v.shape
    w = v.reshape(*lead, two_i // 2, 2, hidden)
    return mx.contiguous(w[..., 0, :]), mx.contiguous(w[..., 1, :])


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        config.vision_config.text_hidden_size = config.text_config.hidden_size
        config.audio_config.text_hidden_size = config.text_config.hidden_size
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)
        self.audio_tower = AudioModel(config.audio_config)

    def get_image_features(self, pixel_values):
        return self.vision_tower(pixel_values)

    def prepare_video_frame_pairs(self, processor, second_frames):
        """Capability hook for the frames fallback (see generate/video.py):
        inkling patches are temporal pairs [P, T=2, H, W, C] and the image
        processor duplicates a still into both slots. The first frame of each
        pair goes through the standard still path; these second frames'
        patches are spliced into temporal slot 1 by ``get_input_embeddings``,
        halving the token cost of independent stills and giving the encoder
        real motion."""
        pixels = processor.image_processor.preprocess(images=second_frames)[
            "pixel_values"
        ]
        return {"video_temporal_pixels": pixels[:, 0]}

    def get_audio_features(self, audio_input_ids, audio_input_ids_mask=None):
        if audio_input_ids_mask is not None:
            flat = audio_input_ids.reshape(-1, audio_input_ids.shape[-1])
            keep = np.nonzero(np.array(audio_input_ids_mask).reshape(-1))[0]
            frames = flat[mx.array(keep)]
        else:
            frames = audio_input_ids.reshape(-1, audio_input_ids.shape[-1])
        return self.audio_tower(frames)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        h = self.language_model.model.embed(input_ids)

        if pixel_values is not None:
            # Native video: pixel_values rows carry a temporal pair per patch
            # [P, T=2, H, W, C], and the image processor duplicates the frame
            # into both slots for stills. A caller that patchified the SECOND
            # frame of each consecutive pair can pass those patches here to
            # overwrite slot 1 of the LAST N rows (video entities are appended
            # after still images), turning duplicated stills into true
            # temporal pairs with no other bookkeeping changes.
            video_slot1 = kwargs.get("video_temporal_pixels", None)
            if video_slot1 is not None:
                n = video_slot1.shape[0]
                head = pixel_values[:-n] if n < pixel_values.shape[0] else None
                tail = pixel_values[-n:]
                tail = mx.stack([tail[:, 0], video_slot1.astype(tail.dtype)], axis=1)
                pixel_values = (
                    tail if head is None else mx.concatenate([head, tail], axis=0)
                )
            feats = self.get_image_features(pixel_values).astype(h.dtype)
            mask = mx.broadcast_to(
                (input_ids == self.config.image_token_id)[..., None], h.shape
            )
            h = masked_scatter(h, mask, feats)

        audio_input_ids = kwargs.get("audio_input_ids", None)
        if audio_input_ids is not None:
            feats = self.get_audio_features(
                audio_input_ids, kwargs.get("audio_input_ids_mask", None)
            ).astype(h.dtype)
            mask = mx.broadcast_to(
                (input_ids == self.config.audio_token_id)[..., None], h.shape
            )
            h = masked_scatter(h, mask, feats)
        return InputEmbeddingsFeatures(inputs_embeds=h)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        **kwargs,
    ):
        spec_kwargs = {
            k: kwargs.pop(k)
            for k in ("return_hidden", "return_shared_kv", "skip_logits")
            if k in kwargs
        }
        embeds = self.get_input_embeddings(
            input_ids, pixel_values=pixel_values, **kwargs
        )
        return self.language_model(
            inputs_embeds=embeds.inputs_embeds, cache=cache, **spec_kwargs
        )

    _ATTN = {
        "wq_du": "q_proj",
        "wk_dv": "k_proj",
        "wv_dv": "v_proj",
        "wr_du": "r_proj",
        "wo_ud": "o_proj",
    }

    def _map_llm_layer(self, base, sub, v):
        out = {}
        if sub.startswith("attn."):
            name, leaf = sub[len("attn.") :].rsplit(".", 1)
            if name in self._ATTN:
                out[base + f"self_attn.{self._ATTN[name]}.weight"] = v
            elif name in ("q_norm", "k_norm"):
                out[base + f"self_attn.{name}.weight"] = v
            elif name in ("k_sconv", "v_sconv"):
                out[base + f"self_attn.{name}.conv.weight"] = v.transpose(0, 2, 1)
            elif name == "rel_logits_proj":
                out[base + "self_attn.rel_proj"] = v
            else:
                out[base + "self_attn." + name + "." + leaf] = v
        elif sub == "attn_norm.weight":
            out[base + "input_layernorm.weight"] = v
        elif sub == "mlp_norm.weight":
            out[base + "post_attention_layernorm.weight"] = v
        elif sub == "attn_sconv.weight":
            out[base + "attn_sconv.conv.weight"] = v.transpose(0, 2, 1)
        elif sub == "mlp_sconv.weight":
            out[base + "mlp_sconv.conv.weight"] = v.transpose(0, 2, 1)
        elif sub.startswith("mlp."):
            m = sub[len("mlp.") :]
            p = base + "mlp."
            if m == "gate.weight":
                out[p + "gate_weight"] = v
            elif m == "gate.bias":
                out[p + "e_score_correction_bias"] = v
            elif m in ("gate.global_scale", "global_scale"):
                out[p + "global_scale"] = v
            elif m == "shared_experts.shared_w13_weight":
                g, u = _split_gate_up(v)
                out[p + "shared_experts.gate_proj.weight"] = g
                out[p + "shared_experts.up_proj.weight"] = u
            elif m == "shared_experts.shared_w2_weight":
                out[p + "shared_experts.down_proj.weight"] = v
            elif m == "w13_dn.weight":
                g, u = _split_gate_up(v)
                out[p + "gate_proj.weight"] = g
                out[p + "up_proj.weight"] = u
            elif m == "w2_md.weight":
                out[p + "down_proj.weight"] = v
            else:
                out[p + m] = v
        else:
            out[base + sub] = v
        return out

    def _map_experts(self, i, buf):
        """Emit MLX-native switch-MLP weights for one MoE layer.

        ``buf`` maps ``"w13"``/``"w2"`` to a dict of the raw checkpoint tensors
        (``weight`` plus, when NVFP4-quantized, ``scale``/``scale2``). NVFP4
        experts are stored as ModelOpt two-level nvfp4: uint8-packed E2M1 codes
        (bit-identical to MLX's uint32 layout via ``view``), per-block E4M3
        ``scale`` (already MLX's ``.scales`` byte layout), and a per-expert
        ``scale2``. We pass the codes and block scales through untouched and
        route ``scale2`` to ``InklingSwitchGLU`` (``gate_scale``/``out_scale``).
        Layer ``dense_mlp_idx`` (the first MoE layer) is excluded from
        quantization and arrives as bf16, handled by the plain-split branch.
        """
        out = {}
        p = f"language_model.model.layers.{i}.mlp.switch_mlp."
        w13, w2 = buf["w13"], buf["w2"]
        quantized = w13["weight"].dtype == mx.uint8
        if quantized:
            gw, uw = _split_gate_up(w13["weight"].view(mx.uint32))
            gs, us = _split_gate_up(w13["scale"])
            out[p + "gate_proj.weight"] = gw
            out[p + "gate_proj.scales"] = gs
            out[p + "up_proj.weight"] = uw
            out[p + "up_proj.scales"] = us
            out[p + "down_proj.weight"] = w2["weight"].view(mx.uint32)
            out[p + "down_proj.scales"] = w2["scale"]
            s13 = w13["scale2"].astype(mx.float32)
            s2 = w2["scale2"].astype(mx.float32)
            out[p + "gate_scale"] = s13
            out[p + "out_scale"] = s13 * s2
        else:
            gw, uw = _split_gate_up(w13["weight"])
            out[p + "gate_proj.weight"] = gw
            out[p + "up_proj.weight"] = uw
            out[p + "down_proj.weight"] = w2["weight"]
            n = gw.shape[0]
            out[p + "gate_scale"] = mx.ones((n,))
            out[p + "out_scale"] = mx.ones((n,))
        return out

    def sanitize(self, weights):
        out = {}
        experts = {}  # layer index -> {"w13"|"w2": {"weight"|"scale"|"scale2": array}}
        for k, v in weights.items():
            if ".mtp" in k or k.startswith("model.mtp") or k.endswith("training_args"):
                continue
            # Routed-expert tensors (base weight + NVFP4 sidecars) are buffered and
            # folded per layer once both w13 and w2 are seen; input_amax /
            # original_shape are calibration metadata with no inference use.
            if ".mlp.experts.w13_weight" in k or ".mlp.experts.w2_weight" in k:
                head, tail = k.split(".mlp.experts.")
                i = head.split("layers.")[1]
                which = "w13" if tail.startswith("w13") else "w2"
                leaf = tail.split("_weight", 1)[1].lstrip(".") or "weight"
                if leaf in ("input_amax", "original_shape"):
                    continue
                experts.setdefault(i, {}).setdefault(which, {})[leaf] = v
                continue
            if k == "model.llm.embed.weight":
                out["language_model.model.embed_tokens.weight"] = v
            elif k == "model.llm.unembed.weight":
                out["language_model.lm_head.weight"] = v
            elif k in ("model.llm.embed_norm.weight", "model.llm.norm.weight"):
                out["language_model.model." + k[len("model.llm.") :]] = v
            elif k.startswith("model.llm.layers."):
                i, sub = k[len("model.llm.layers.") :].split(".", 1)
                out.update(
                    self._map_llm_layer(f"language_model.model.layers.{i}.", sub, v)
                )
            elif k.startswith("model.visual."):
                sub = k[len("model.visual.") :]
                if sub.startswith("layers.linear_"):
                    j = sub[len("layers.linear_") :].split(".")[0]
                    out[f"vision_tower.encoder_layers.{j}.projection.weight"] = v
                elif sub.startswith("layers.norm_"):
                    j = sub[len("layers.norm_") :].split(".")[0]
                    out[f"vision_tower.encoder_layers.{j}.layer_norm.weight"] = v
                else:
                    out["vision_tower." + sub] = v
            elif k.startswith("model.audio."):
                sub = k[len("model.audio.") :]
                if sub == "encoder.weight":
                    out["audio_tower.embed_audio_tokens.weight"] = v
                elif sub == "final_norm.weight":
                    out["audio_tower.norm.weight"] = v
                else:
                    out["audio_tower." + sub] = v
            else:
                out[k] = v
        for i, buf in experts.items():
            out.update(self._map_experts(i, buf))
        return fuse_qkvr(shared_experts_to_dense(out))

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.model.layers
