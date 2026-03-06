"""
Processor compatibility patch for Phi4-SigLIP in mlx-vlm.

The HuggingFace Phi4VisionRProcessor uses custom classes from the model repo
(Siglip2ImageProcessorNoUpscale, tokenizer_image_token) that may return
PyTorch tensors. This patch ensures:
1. Image processing returns numpy arrays
2. The <image> token is handled correctly with IMAGE_TOKEN_INDEX = -200
3. spatial_shapes are properly converted for mlx-vlm
"""

import numpy as np
from transformers import AutoProcessor

from ..base import install_auto_processor_patch

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """
    Tokenize a prompt containing <image> tokens.
    Replaces <image> with IMAGE_TOKEN_INDEX in the token sequence.
    """
    prompt_chunks = [
        tokenizer(chunk, add_special_tokens=False).input_ids
        for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    return input_ids


class Phi4SigLipProcessor:
    """
    Processor for Phi4-SigLIP that wraps the HuggingFace processor
    and ensures numpy/mlx compatibility.
    """

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text=None,
        images=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        from transformers import BatchFeature

        data = {}

        # Process images
        if images is not None:
            if not isinstance(images, list):
                images = [images]

            # Use the image processor (Siglip2ImageProcessor or similar)
            # Request numpy return to avoid PyTorch dependency
            image_outputs = self.image_processor(images, return_tensors="np")

            # Extract and convert outputs
            if "pixel_values" in image_outputs:
                pv = image_outputs["pixel_values"]
                if hasattr(pv, "numpy"):
                    pv = pv.numpy()
                data["pixel_values"] = np.array(pv)

            if "pixel_attention_mask" in image_outputs:
                pam = image_outputs["pixel_attention_mask"]
                if hasattr(pam, "numpy"):
                    pam = pam.numpy()
                data["pixel_attention_mask"] = np.array(pam)

            if "spatial_shapes" in image_outputs:
                ss = image_outputs["spatial_shapes"]
                if hasattr(ss, "numpy"):
                    ss = ss.numpy()
                elif isinstance(ss, list):
                    ss = np.array(ss)
                data["spatial_shapes"] = np.array(ss)

        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]

            # Apply chat template if not already applied
            processed_text = []
            for t in text:
                if "<|im_start|>" not in t and self.tokenizer.chat_template:
                    messages = [{"role": "user", "content": t}]
                    t = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                processed_text.append(t)
            text = processed_text

            has_images = any(DEFAULT_IMAGE_TOKEN in t for t in text)

            if has_images and images is not None:
                # Tokenize with image token handling
                input_ids_list = []
                for t in text:
                    ids = tokenizer_image_token(t, self.tokenizer)
                    input_ids_list.append(ids)

                # Pad if needed
                if padding and len(input_ids_list) > 1:
                    max_len = max(len(ids) for ids in input_ids_list)
                    pad_token_id = self.tokenizer.pad_token_id or 0
                    padded_ids = []
                    attention_masks = []
                    for ids in input_ids_list:
                        pad_len = max_len - len(ids)
                        padded_ids.append(ids + [pad_token_id] * pad_len)
                        attention_masks.append([1] * len(ids) + [0] * pad_len)
                    data["input_ids"] = np.array(padded_ids)
                    data["attention_mask"] = np.array(attention_masks)
                else:
                    data["input_ids"] = np.array([input_ids_list[0]])
                    data["attention_mask"] = np.ones_like(data["input_ids"])
            else:
                text_inputs = self.tokenizer(
                    text,
                    padding=padding,
                    return_tensors="np",
                )
                data["input_ids"] = np.array(text_inputs["input_ids"])
                data["attention_mask"] = np.array(text_inputs["attention_mask"])

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @property
    def model_input_names(self):
        tokenizer_names = self.tokenizer.model_input_names
        image_names = getattr(
            self.image_processor, "model_input_names", ["pixel_values"]
        )
        return list(dict.fromkeys(tokenizer_names + image_names))

    def save_pretrained(self, save_directory, **kwargs):
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        if hasattr(self.image_processor, "save_pretrained"):
            self.image_processor.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Load config to determine vision tower and create image processor
        model_path = Path(pretrained_model_name_or_path)

        config = {}
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            try:
                from huggingface_hub import hf_hub_download

                cfg_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
                with open(cfg_path) as f:
                    config = json.load(f)
            except Exception:
                pass

        vision_tower_name = config.get("mm_vision_tower", "")
        vision_config = config.get("vision_config", {})
        max_num_patches = config.get("max_num_patches", 3600)
        min_num_patches = config.get("min_num_patches", 256)

        # Determine patch_size
        patch_size = vision_config.get("patch_size", None)
        if patch_size is None:
            if "patch14" in vision_tower_name.lower():
                patch_size = 14
            else:
                patch_size = 16

        # Create NaFlex-compatible image processor
        if "naflex" in vision_tower_name.lower():
            try:
                from transformers.models.siglip2.image_processing_siglip2 import (
                    Siglip2ImageProcessor,
                )

                image_processor = Siglip2ImageProcessor(
                    patch_size=patch_size,
                    max_num_patches=max_num_patches,
                    min_num_patches=min_num_patches,
                    do_resize=True,
                    do_rescale=True,
                    do_normalize=True,
                    image_mean=[0.5, 0.5, 0.5],
                    image_std=[0.5, 0.5, 0.5],
                )
            except ImportError:
                # Fallback: try loading the custom processor from model repo
                image_processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path, **kwargs
                ).image_processor
        else:
            from transformers import SiglipImageProcessor

            img_size = vision_config.get("image_size", 384)
            image_processor = SiglipImageProcessor(
                size={"height": img_size, "width": img_size}
            )

        return cls(image_processor=image_processor, tokenizer=tokenizer)


# Register processor patch so AutoProcessor returns our custom processor
install_auto_processor_patch(["phi4-siglip"], Phi4SigLipProcessor)
