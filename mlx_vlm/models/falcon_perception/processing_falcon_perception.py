import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

from ..base import install_auto_processor_patch

_IMAGE_MEAN = [0.5, 0.5, 0.5]
_IMAGE_STD = [0.5, 0.5, 0.5]


def _resize_if_necessary(
    image: Image.Image,
    shortest: int = 256,
    longest: int = 1024,
) -> Image.Image:
    w, h = image.size
    ar = w / h
    if shortest <= w <= longest and shortest <= h <= longest:
        return image
    is_vert = w < h
    if w < shortest or h < shortest:
        if is_vert:
            new_w, new_h = shortest, int(shortest / ar)
        else:
            new_h, new_w = shortest, int(shortest * ar)
    else:
        if is_vert:
            new_w = longest
            new_h = int(new_w / ar)
        else:
            new_h = longest
            new_w = int(new_h * ar)
    if new_w > longest:
        new_w = longest
        new_h = int(new_w / ar)
    if new_h > longest:
        new_h = longest
        new_w = int(new_h * ar)
    return image.resize((new_w, new_h), Image.BICUBIC)


def _smart_resize(
    image: Image.Image,
    factor: int,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> Image.Image:
    w, h = image.size

    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor

    if (w_bar, h_bar) != (w, h):
        image = image.resize((w_bar, h_bar), Image.BICUBIC)

    return image


def preprocess_image(
    pil_image: Image.Image,
    spatial_patch_size: int = 16,
    min_image_size: int = 256,
    max_image_size: int = 1024,
) -> tuple:
    pil_image = _resize_if_necessary(pil_image, min_image_size, max_image_size)
    pil_image = pil_image.convert("RGB")
    pil_image = _smart_resize(pil_image, factor=spatial_patch_size)

    img_np = np.array(pil_image).astype(np.float32) / 255.0

    mean = np.array(_IMAGE_MEAN, dtype=np.float32)
    std = np.array(_IMAGE_STD, dtype=np.float32)
    img_np = (img_np - mean) / std

    h, w, _ = img_np.shape
    grid_h = h // spatial_patch_size
    grid_w = w // spatial_patch_size
    return img_np, grid_h, grid_w


class FalconPerceptionProcessor:
    def __init__(self, tokenizer, config: dict):
        self.tokenizer = tokenizer
        self._config = config

        self.spatial_patch_size = config.get("spatial_patch_size", 16)
        self.img_id = config.get("img_id", 227)
        self.img_end_id = config.get("img_end_id", 230)
        self.image_cls_token_id = config.get("image_cls_token_id", 244)
        self.image_reg_1_token_id = config.get("image_reg_1_token_id", 245)
        self.image_reg_2_token_id = config.get("image_reg_2_token_id", 246)
        self.image_reg_3_token_id = config.get("image_reg_3_token_id", 247)
        self.image_reg_4_token_id = config.get("image_reg_4_token_id", 248)

        self._image_prefix_ids = [
            self.image_cls_token_id,
            self.image_reg_1_token_id,
            self.image_reg_2_token_id,
            self.image_reg_3_token_id,
            self.image_reg_4_token_id,
        ]

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("use_fast", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()

        if is_local:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            config_file = model_path / "config.json"
            config = json.loads(config_file.read_text()) if config_file.exists() else {}
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )
            try:
                from huggingface_hub import hf_hub_download

                cfg_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
                config = json.loads(Path(cfg_path).read_text())
            except Exception:
                config = {}

        return cls(tokenizer, config)

    def apply_chat_template(self, messages=None, *args, **kwargs):
        if messages is not None:
            wrapped = []
            for m in messages:
                if isinstance(m, str):
                    wrapped.append({"role": "user", "content": self._make_prompt(m)})
                elif isinstance(m, dict) and m.get("role") == "user":
                    wrapped.append(
                        {**m, "content": self._make_prompt(m.get("content", ""))}
                    )
                else:
                    wrapped.append(m)
            messages = wrapped
        return self.tokenizer.apply_chat_template(messages, *args, **kwargs)

    def _make_prompt(self, query: str) -> str:
        return (
            f"<|image|>Segment these expressions in the image:"
            f"<|start_of_query|>{query}<|REF_SEG|>"
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def __call__(
        self,
        text=None,
        images=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]

        if images is None or (isinstance(images, list) and len(images) == 0):
            text_inputs = self.tokenizer(text, padding=padding, **kwargs)
            return BatchFeature(data=dict(text_inputs), tensor_type=return_tensors)

        if not isinstance(images, list):
            images = [images]

        text = [self._make_prompt(t) for t in text]

        pixel_list = []
        grid_hws = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img = img.convert("RGB")
            pv, gh, gw = preprocess_image(img, self.spatial_patch_size)
            pixel_list.append(pv)
            grid_hws.append([gh, gw])

        all_ids = []
        img_offset = 0
        for t in text:
            token_ids = self.tokenizer.encode(t, add_special_tokens=False)
            n_img = sum(1 for tid in token_ids if tid == self.img_id)
            sample_grids = grid_hws[img_offset : img_offset + n_img]
            all_ids.append(self._expand_image_tokens(token_ids, sample_grids))
            img_offset += n_img

        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        attention_masks = []
        for ids in all_ids:
            pad_len = max_len - len(ids) if padding else 0
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        data = {
            "input_ids": padded_ids,
            "attention_mask": attention_masks,
            "pixel_values": np.stack(pixel_list),
            "image_grid_hw": np.array(grid_hws, dtype=np.int32),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def process(
        self,
        text: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        padding: bool = True,
        return_tensors: str = "mlx",
        **kwargs,
    ) -> Dict[str, Any]:
        result = self(
            text=text,
            images=images,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs,
        )
        return dict(result)

    def _expand_image_tokens(
        self, token_ids: List[int], grid_hws: List[List[int]]
    ) -> List[int]:
        expanded = []
        img_idx = 0

        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.img_id and img_idx < len(grid_hws):
                gh, gw = grid_hws[img_idx]
                n_patches = gh * gw
                expanded.extend(self._image_prefix_ids)
                expanded.extend([self.img_id] * n_patches)
                expanded.append(self.img_end_id)
                img_idx += 1
            else:
                expanded.append(token_ids[i])
            i += 1

        return expanded


def generate_perception(
    model,
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
    import mlx.core as mx

    from mlx_vlm.utils import load_image as _load_image

    if not isinstance(image, Image.Image):
        image = _load_image(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    from ..base import to_mlx

    result = processor(text=[query], images=[image], padding=False)
    result = to_mlx(result)
    input_ids = result["input_ids"]
    pixel_values = result["pixel_values"]
    image_grid_hw = result.get("image_grid_hw", None)

    config = model.config
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

    from ..cache import make_prompt_cache

    cache = make_prompt_cache(model)

    # Prefill: run full model with image
    logits_out = model(
        input_ids,
        pixel_values=pixel_values,
        cache=cache,
        image_grid_hw=image_grid_hw,
    )
    logits = logits_out.logits

    # Compute segmentation features from prefill hidden states (via AnyUp if available)
    segm_features = None
    if hasattr(model, "conv_segm"):
        prefill_hidden = model.last_hidden_state
        if prefill_hidden is not None:
            segm_features = model.compute_segm_features(
                prefill_hidden, input_ids, pixel_values, grid_h, grid_w
            )
            mx.eval(segm_features)

    # After prefill, clear _position_ids so decode falls back to _rope_delta.
    # _position_ids is sized for prefill only; indexing it at decode offsets fails.
    lm = model.language_model
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
        h_state = model.last_hidden_state
        h_last = h_state[:, -1:, :] if h_state is not None else None

        if token_id == coord_token_id and h_last is not None:
            coord_logits = model.decode_coords(h_last.reshape(-1, h_last.shape[-1]))
            num_bins = coord_logits.shape[-1]
            pred_bins = mx.argmax(coord_logits, axis=-1)
            pred_x = pred_bins[0, 0].item() / (num_bins - 1)
            pred_y = pred_bins[0, 1].item() / (num_bins - 1)
            coord_xy = mx.array([[pred_x, pred_y]])
            current_det["xy"] = {"x": pred_x, "y": pred_y}

        elif token_id == size_token_id and h_last is not None:
            size_logits = model.decode_sizes(h_last.reshape(-1, h_last.shape[-1]))
            hw_pred = model.process_sizes(size_logits)
            pred_h = hw_pred[0, 0].item()
            pred_w = hw_pred[0, 1].item()
            size_hw_val = mx.array([[pred_h, pred_w]])
            current_det["hw"] = {"h": pred_h, "w": pred_w}

        elif token_id == seg_token_id and h_last is not None:
            if segm_features is not None:
                seg_hidden = h_last.reshape(-1)  # (D,)
                mask = model.decode_segm_mask(
                    seg_hidden, segm_features, orig_h, orig_w, segm_threshold
                )
                mx.eval(mask)
                current_det["mask"] = mask
            if "xy" in current_det and "hw" in current_det:
                detections.append(current_det)
            current_det = {}

        # Decode step: embed token, apply coord/size encoding, run LM
        embeds = embed_fn(token_2d)
        embeds = model.encode_coords_into_embeds(
            embeds,
            token_2d,
            coord_xy if token_id == coord_token_id else None,
        )
        embeds = model.encode_sizes_into_embeds(
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


def plot_detections(image, detections, save_path="perception_output.png"):
    """Plot bounding box detections and segmentation masks on the image and save."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if not isinstance(image, Image.Image):
        from mlx_vlm.utils import load_image as _load_image

        image = _load_image(image)
    image = image.convert("RGB")

    w, h = image.size
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(detections), 1)))

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        xy = det["xy"]
        hw = det["hw"]

        # Draw segmentation mask if available
        if "mask" in det:
            import mlx.core as mx
            from PIL import Image as _PILImage

            mask_arr = det["mask"]
            if isinstance(mask_arr, mx.array):
                mask_arr = np.array(mask_arr)
            # Resize mask to match image if needed
            if mask_arr.shape != (h, w):
                mask_arr = np.array(
                    _PILImage.fromarray(mask_arr.astype(np.uint8)).resize(
                        (w, h), _PILImage.NEAREST
                    )
                ).astype(bool)
            mask_rgba = np.zeros((h, w, 4))
            mask_rgba[mask_arr > 0] = [*color[:3], 0.4]
            ax.imshow(mask_rgba)

        # Draw bounding box
        cx = xy["x"] * w
        cy = xy["y"] * h
        bw = hw["w"] * w
        bh = hw["h"] * h

        x0 = cx - bw / 2
        y0 = cy - bh / 2

        rect = patches.Rectangle(
            (x0, y0),
            bw,
            bh,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 5,
            f"det {i}",
            color=color,
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


install_auto_processor_patch("falcon_perception", FalconPerceptionProcessor)

__all__ = [
    "FalconPerceptionProcessor",
    "generate_perception",
    "plot_detections",
]
