"""Processor for MolmoPoint - no torch dependency."""

import numpy as np

from ..base import install_auto_processor_patch, load_chat_template, to_mlx
from .image_processing import preprocess_images

# Special tokens
IMAGE_PATCH_TOKEN = "<im_patch>"
IMAGE_LOW_RES_TOKEN = "<im_low>"
IM_START_TOKEN = "<im_start>"
LOW_RES_IMAGE_START_TOKEN = "<low_res_im_start>"
IM_END_TOKEN = "<im_end>"
IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"

IMAGE_TOKENS = [
    IMAGE_PATCH_TOKEN,
    IMAGE_LOW_RES_TOKEN,
    IM_COL_TOKEN,
    IM_START_TOKEN,
    LOW_RES_IMAGE_START_TOKEN,
    IM_END_TOKEN,
]


class MolmoPointProcessor:
    """Processor for MolmoPoint that handles image processing and tokenization."""

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.image_processor = None  # Not a BaseImageProcessor
        # Defaults from processor_config.json
        self.image_use_col_tokens = kwargs.get("image_use_col_tokens", True)
        self.use_single_crop_col_tokens = kwargs.get(
            "use_single_crop_col_tokens", False
        )
        self.use_single_crop_start_token = kwargs.get(
            "use_single_crop_start_token", True
        )
        self.use_low_res_token_for_global_crops = kwargs.get(
            "use_low_res_token_for_global_crops", True
        )

        self.image_token_ids = [
            tokenizer.convert_tokens_to_ids(token) for token in IMAGE_TOKENS
        ]

    def save_pretrained(self, save_directory):
        """Save the tokenizer to the given directory."""
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        return cls(tokenizer)

    def get_image_tokens(self, image_grid):
        resized_h, resized_w, height, width = image_grid
        per_row = np.full(width, IMAGE_PATCH_TOKEN)
        if self.image_use_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [IM_START_TOKEN],
            np.tile(per_row, [height]),
            [IM_END_TOKEN],
        ]
        if self.use_low_res_token_for_global_crops:
            per_row = np.full(resized_w, IMAGE_LOW_RES_TOKEN)
        else:
            per_row = np.full(resized_w, IMAGE_PATCH_TOKEN)
        use_single_crop_col_tokens = (
            self.image_use_col_tokens
            if self.use_single_crop_col_tokens is None
            else self.use_single_crop_col_tokens
        )
        image_start_token = (
            LOW_RES_IMAGE_START_TOKEN
            if self.use_single_crop_start_token
            else IM_START_TOKEN
        )
        if use_single_crop_col_tokens:
            per_row = np.concatenate([per_row, [IM_COL_TOKEN]], 0)
        joint = [
            [image_start_token],
            np.tile(per_row, [resized_h]),
            [IM_END_TOKEN],
        ] + joint
        return np.concatenate(joint)

    def __call__(
        self,
        text=None,
        images=None,
        padding=True,
        return_tensors="mlx",
        **kwargs,
    ):
        """Process text and images for MolmoPoint."""
        if images is not None:
            if not isinstance(images, list):
                images = [images]

            # Process images (settings from preprocessor_config.json)
            img_result = preprocess_images(
                images,
                max_crops=24,
                overlap_margins=[4, 4],
                base_image_input_size=(378, 378),
                image_patch_size=14,
                pooling_size=[2, 2],
                return_pointing_metadata=True,
            )

            image_grids = img_result["image_grids"]

            # Insert image tokens into text
            if not isinstance(text, list):
                text = [text]

            text = [t for t in text]  # copy
            index = 0
            for i in range(len(text)):
                num_images = text[i].count(IMAGE_PROMPT)
                for grid in image_grids[index : index + num_images]:
                    image_tokens = self.get_image_tokens(grid)
                    image_string = "".join(image_tokens)
                    text[i] = text[i].replace(IMAGE_PROMPT, image_string, 1)
                index += num_images

            # Tokenize
            text_inputs = self.tokenizer(
                text,
                padding=padding,
                return_tensors="np" if return_tensors == "mlx" else return_tensors,
            )

            input_ids = np.array(text_inputs["input_ids"])
            attention_mask = np.array(text_inputs["attention_mask"])

            # Insert BOS if needed
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            if bos is not None and input_ids.shape[1] > 0:
                if len(input_ids.shape) == 1:
                    input_ids = input_ids[None, :]
                    attention_mask = attention_mask[None, :]
                first_valid = (attention_mask == 1).argmax(axis=-1)
                if not np.all(
                    input_ids[np.arange(input_ids.shape[0]), first_valid] == bos
                ):
                    B, S = input_ids.shape
                    new_ids = np.full(
                        (B, S + 1),
                        self.tokenizer.pad_token_id or 0,
                        dtype=input_ids.dtype,
                    )
                    new_mask = np.zeros((B, S + 1), dtype=attention_mask.dtype)
                    for b in range(B):
                        fv = int(first_valid[b])
                        new_ids[b, fv] = bos
                        new_mask[b, fv] = 1
                        new_ids[b, fv + 1 : fv + 1 + S - fv] = input_ids[b, fv:]
                        new_mask[b, fv + 1 : fv + 1 + S - fv] = attention_mask[b, fv:]
                    input_ids = new_ids
                    attention_mask = new_mask

            # Build token_type_ids for bidirectional attention on image tokens
            image_token_ids_arr = np.array(self.image_token_ids).astype(input_ids.dtype)
            token_type_ids = np.any(
                input_ids[:, :, None] == image_token_ids_arr[None, None, :], axis=-1
            )

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids.astype(np.int64),
                "pixel_values": img_result["pixel_values"],
                "image_token_pooling": img_result["image_token_pooling"],
                "image_grids": img_result["image_grids"],
                "image_num_crops": img_result["image_num_crops"],
            }

            # Store pointing metadata on the processor for later point extraction
            if "_pointing_metadata" in img_result:
                self._pointing_metadata = img_result["_pointing_metadata"]

            if return_tensors == "mlx":
                return to_mlx(result)
            return result
        else:
            # Text-only
            tokens = self.tokenizer(
                text,
                padding=padding,
                return_tensors="np" if return_tensors == "mlx" else return_tensors,
            )
            if return_tensors == "mlx":
                return to_mlx(dict(tokens))
            return dict(tokens)


install_auto_processor_patch("molmo_point", MolmoPointProcessor)
