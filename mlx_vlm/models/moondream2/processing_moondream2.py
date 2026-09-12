import numpy as np
from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch
from .image_crops import create_crops


class Moondream2Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(PreTrainedTokenizerFast.from_pretrained(path, **kwargs))

    def __call__(
        self,
        text=None,
        images=None,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        return_tensors=None,
        **kwargs,
    ):
        texts = [text] if isinstance(text, str) else list(text or [])
        images = (
            list(images)
            if isinstance(images, (list, tuple))
            else ([] if images is None else [images])
        )
        if not texts:
            raise ValueError("Moondream2 requires a text prompt")
        if images and len(texts) != 1 and len(images) != len(texts):
            raise ValueError("Multiple prompts require one image per prompt")
        result = {}
        if images:
            crops, counts, layouts = [], [], []
            for image in images:
                image_crops, layout = create_crops(image, 378, 12, 4)
                crops.extend(image_crops)
                counts.append(len(image_crops))
                layouts.append(layout)
            result.update(
                pixel_values=np.stack(crops), num_crops=counts, crop_layouts=layouts
            )
        sequences = []
        per_prompt = len(images) if len(texts) == 1 else 1
        for prompt in texts:
            if images:
                query = (
                    [198, 198, 24361, 25]
                    + self.tokenizer.encode(" " + prompt, add_special_tokens=False)
                    + [198, 198, 33706, 25]
                )
                ids = [self.tokenizer.bos_token_id] + [0] * (729 * per_prompt) + query
            else:
                ids = self.tokenizer.encode(
                    prompt, add_special_tokens=add_special_tokens
                )
            sequences.append(ids)
        width = max(map(len, sequences)) if padding else None
        masks = []
        for i, ids in enumerate(sequences):
            count = width - len(ids) if width is not None else 0
            pad = [self.tokenizer.pad_token_id or self.tokenizer.eos_token_id] * count
            if padding_side == "left":
                sequences[i] = pad + ids
                masks.append([0] * count + [1] * len(ids))
            elif padding_side == "right":
                sequences[i] = ids + pad
                masks.append([1] * len(ids) + [0] * count)
            else:
                raise ValueError("padding_side must be left or right")
        result.update(
            input_ids=np.array(sequences, dtype=np.int32),
            attention_mask=np.array(masks, dtype=np.int32),
        )
        return result


install_auto_processor_patch("moondream1", Moondream2Processor)
