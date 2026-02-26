import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin


class MiniCPMOImageProcessor:
    def __init__(
        self,
        size: Union[int, tuple] = 448,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    ):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(self, images: Union[Image.Image, List[Image.Image]]):
        if isinstance(images, Image.Image):
            images = [images]

        out = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((self.size[1], self.size[0]), Image.Resampling.BICUBIC)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.array(self.image_std, dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
            out.append(arr)
        return out


class MiniCPMOProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: MiniCPMOImageProcessor,
        tokenizer,
        image_feature_size: int = 64,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_feature_size = image_feature_size
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.image_pad_token = "<|image_pad|>"
        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, model_path_or_repo: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_repo,
            trust_remote_code=True,
            use_fast=True,
        )

        model_path = Path(model_path_or_repo)
        image_size = 448
        image_feature_size = 64
        image_mean = (0.5, 0.5, 0.5)
        image_std = (0.5, 0.5, 0.5)

        try:
            cfg_path = model_path / "config.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                image_size = cfg.get("vision_config", {}).get(
                    "image_size", cfg.get("image_size", 448)
                )
                image_feature_size = int(cfg.get("query_num", 64))
        except Exception:
            pass

        try:
            pp_path = model_path / "preprocessor_config.json"
            if pp_path.exists():
                with open(pp_path, "r", encoding="utf-8") as f:
                    pp = json.load(f)
                if "image_mean" in pp:
                    image_mean = tuple(pp["image_mean"])
                if "image_std" in pp:
                    image_std = tuple(pp["image_std"])
                if "size" in pp:
                    s = pp["size"]
                    if isinstance(s, dict):
                        pp_size = int(s.get("height", s.get("shortest_edge", image_size)))
                        image_size = max(image_size, pp_size)
                    elif isinstance(s, int):
                        image_size = max(image_size, s)
        except Exception:
            pass

        image_processor = MiniCPMOImageProcessor(
            size=image_size,
            image_mean=image_mean,
            image_std=image_std,
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_feature_size=image_feature_size,
        )

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("tokenize", False)
        if not args:
            return self.tokenizer.apply_chat_template(*args, **kwargs)

        messages = args[0]
        normalized_messages = []
        image_span = (
            self.vision_start_token
            + (self.image_pad_token * self.image_feature_size)
            + self.vision_end_token
        )
        has_visual_placeholder = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        parts.append(str(item))
                        continue
                    item_type = item.get("type")
                    if item_type == "image":
                        parts.append(image_span)
                        has_visual_placeholder = True
                    elif item_type == "text":
                        parts.append(item.get("text", ""))
                content = "\n".join([p for p in parts if p])
            elif isinstance(content, str) and self.image_pad_token in content:
                has_visual_placeholder = True

            normalized_messages.append(
                {
                    "role": msg.get("role", "user"),
                    "content": content if isinstance(content, str) else str(content),
                }
            )

        if not has_visual_placeholder:
            for idx in range(len(normalized_messages) - 1, -1, -1):
                if normalized_messages[idx].get("role") == "user":
                    content = normalized_messages[idx].get("content", "")
                    normalized_messages[idx]["content"] = (
                        f"{image_span}\n{content}" if content else image_span
                    )
                    break

        new_args = (normalized_messages,) + args[1:]
        return self.tokenizer.apply_chat_template(*new_args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if text is None:
            text = ""
        if isinstance(text, str):
            text = [text]

        tok_kwargs = {
            "padding": padding,
            "add_special_tokens": add_special_tokens,
            **{k: v for k, v in kwargs.items() if k in {"truncation", "max_length"}},
        }
        if return_tensors is not None:
            tok_kwargs["return_tensors"] = return_tensors

        tok = self.tokenizer(text, **tok_kwargs)

        if images is None:
            return tok

        out = {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }

        if isinstance(images, Image.Image):
            images = [images]
        out["pixel_values"] = np.stack(self.image_processor.preprocess(images), axis=0)

        return out
