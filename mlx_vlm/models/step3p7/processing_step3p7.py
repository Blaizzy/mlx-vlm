from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin

from ...tokenizer_utils import BPEStreamingDetokenizer

MAX_IMAGE_SIZE = 3024


def _patch_step_byte_level_decoder(tokenizer):
    vocab = getattr(tokenizer, "vocab", None)
    if vocab is None and hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None

    if not isinstance(vocab, dict) or not any(
        isinstance(token, str) and token.startswith(("Ġ", "Ċ")) for token in vocab
    ):
        return tokenizer

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is None:
        return tokenizer

    try:
        from tokenizers import decoders

        backend_tokenizer.decoder = decoders.ByteLevel()
    except Exception:
        pass

    return tokenizer


class ImagePatcher:
    def determine_window_size(self, long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def _preprocess_size(self, w: int, h: int) -> tuple[int, int]:
        if max(w, h) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(w, h)
            w = int(w * scale)
            h = int(h * scale)
        return w, h

    def _crop_size(self, w: int, h: int, window: int) -> tuple[int, int]:
        wr, hr = w / window, h / window
        nw = w if wr < 1 else window * (int(wr) + (1 if wr - int(wr) > 0.2 else 0))
        nh = h if hr < 1 else window * (int(hr) + (1 if hr - int(hr) > 0.2 else 0))
        return int(nw), int(nh)

    def _windows(self, w: int, h: int, window: int):
        x_num = 1 if w <= window else int(np.ceil((w - window) / window + 1))
        y_num = 1 if h <= window else int(np.ceil((h - window) / window + 1))
        xs = [window * i for i in range(x_num)]
        ys = [window * i for i in range(y_num)]
        if len(xs) > 1 and xs[-1] + window > w:
            xs[-1] = w - window
        if len(ys) > 1 and ys[-1] + window > h:
            ys[-1] = h - window
        return [(x, y, window, window) for y in ys for x in xs], x_num

    def get_num_patches(self, w: int, h: int) -> tuple[int, int]:
        if w != h:
            size = max(w, h)
            w = h = size
        w, h = self._preprocess_size(w, h)
        window = self.determine_window_size(max(h, w), min(h, w))
        if window == 0:
            return 0, 0
        w, h = self._crop_size(w, h, window)
        windows, x_num = self._windows(w, h, window)
        rows = (len(windows) - 1) // x_num + 1
        if windows and len(windows) % x_num == 0:
            rows -= 1
        return len(windows), rows

    def __call__(self, img: Image.Image):
        img = ImageOps.exif_transpose(img.convert("RGB"))
        w, h = img.size
        if w != h:
            img = self.square_pad(img)
            w, h = img.size
        w, h = self._preprocess_size(w, h)
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        window = self.determine_window_size(max(h, w), min(h, w))
        if window == 0:
            return img, [], None
        crop_w, crop_h = self._crop_size(w, h, window)
        crop_img = img.resize((crop_w, crop_h), Image.Resampling.BILINEAR)
        windows, x_num = self._windows(crop_w, crop_h, window)
        patches = [crop_img.crop((x, y, x + pw, y + ph)) for x, y, pw, ph in windows]
        newlines = [(i + 1) % x_num == 0 for i in range(len(patches))]
        if newlines and newlines[-1]:
            newlines[-1] = False
        return img, patches, newlines if patches else None


class Step3VLProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"
    detokenizer_class = BPEStreamingDetokenizer

    @property
    def detokenizer(self):
        detokenizer = getattr(self, "_detokenizer", None)
        tokenizer = getattr(self, "tokenizer", None)
        if detokenizer is None and tokenizer is not None:
            detokenizer = self.detokenizer_class(tokenizer)
            self._detokenizer = detokenizer
        return detokenizer

    @detokenizer.setter
    def detokenizer(self, _detokenizer):
        tokenizer = getattr(self, "tokenizer", None)
        self._detokenizer = (
            self.detokenizer_class(tokenizer) if tokenizer is not None else _detokenizer
        )

    def __init__(self, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)
        self.image_size = 728
        self.patch_size = 504
        self.num_image_feature_size = 169
        self.num_patch_feature_size = 81
        self.image_token = "<im_patch>"
        self.patcher = ImagePatcher()
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.image_token]

    def _tensor(self, image: Image.Image, size: int):
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr.transpose(2, 0, 1)[None, ...]

    def _patch_repl(self, num_patches: int, newline_mask: Optional[list[bool]]):
        patch_start = self.tokenizer.convert_tokens_to_ids("<patch_start>")
        patch_end = self.tokenizer.convert_tokens_to_ids("<patch_end>")
        patch_newline = self.tokenizer.convert_tokens_to_ids("<patch_newline>")
        patch_body = [self.image_token_id] * self.num_patch_feature_size
        ids = []
        for i in range(num_patches):
            ids.append(patch_start)
            ids.extend(patch_body)
            ids.append(patch_end)
            if newline_mask and newline_mask[i]:
                ids.append(patch_newline)
        return ids

    def _image_repl(self):
        return (
            [self.tokenizer.convert_tokens_to_ids("<im_start>")]
            + [self.image_token_id] * self.num_image_feature_size
            + [self.tokenizer.convert_tokens_to_ids("<im_end>")]
        )

    def _normalize_images(self, images):
        if images is None:
            return []
        if isinstance(images, Image.Image):
            return [images]
        if not isinstance(images, list):
            return [Image.open(images)]
        if images and isinstance(images[0], list):
            images = images[0]
        return [
            img if isinstance(img, Image.Image) else Image.open(img) for img in images
        ]

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: ImageInput | None = None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        texts = [text or ""] if not isinstance(text, list) else text
        images = self._normalize_images(images)
        image_inputs = {}
        if images:
            pixel_values, patch_values, num_patches = [], [], []
            replacement_ids = []
            image_repl = self._image_repl()
            for image in images:
                base, patches, newline_mask = self.patcher(image)
                pixel_values.append(self._tensor(base, self.image_size))
                patch_values.extend(self._tensor(p, self.patch_size) for p in patches)
                num_patches.append(len(patches))
                replacement_ids.extend(self._patch_repl(len(patches), newline_mask))
                replacement_ids.extend(image_repl)
            image_inputs["pixel_values"] = np.concatenate(pixel_values, axis=0)
            image_inputs["num_patches"] = num_patches
            if patch_values:
                image_inputs["patch_pixel_values"] = np.concatenate(
                    patch_values, axis=0
                )
            encoded = []
            for t in texts:
                chunks = t.split(self.image_token)
                if len(chunks) != 2:
                    raise ValueError(
                        "Step-3.7 image prompts must contain one <im_patch> placeholder per image."
                    )
                ids = self.tokenizer(chunks[0], add_special_tokens=False).input_ids
                ids.extend(replacement_ids)
                ids.extend(
                    self.tokenizer(chunks[1], add_special_tokens=False).input_ids
                )
                encoded.append(ids)
            max_len = max(len(ids) for ids in encoded)
            pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            input_ids = np.array(
                [ids + [pad] * (max_len - len(ids)) for ids in encoded]
            )
            attention_mask = (input_ids != pad).astype(np.int32)
            text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            text_inputs = self.tokenizer(texts)
        return BatchFeature({**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        kwargs.setdefault("fix_mistral_regex", True)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        _patch_step_byte_level_decoder(tokenizer)
        return cls(
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


__all__ = ["Step3VLProcessor"]
