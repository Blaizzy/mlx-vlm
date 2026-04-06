import io
import json
import math
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
from transformers import PreTrainedTokenizer

try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.jsonl"}

INVALID_SCORE = -20000000
UNKNOWN_SCORE = -10000000

TABLE_PIECE_LENGTH = 0
TABLE_TOKEN_ID = 1
TABLE_SCORE = 2
TABLE_PIECE_ID = 3

PATH_TOKEN_LENGTH = 0
PATH_TOKEN_ID = 1
PATH_NUM_TOKENS = 2


class AhoCorasick:
    def __init__(self) -> None:
        self._tokens: list[str] = []
        self._bytes = np.zeros(256, dtype=np.int32)
        self._to_suffix_id: dict[int, int] = {}
        self._table = np.zeros((0, 4), dtype=np.int32)

    def build(self, vocab: list[Any]) -> None:
        self._bytes = np.zeros(256, dtype=np.int32)
        self._to_suffix_id = {}

        suffix_to_score: dict[str, float] = {}
        token_to_token_id: dict[str, int] = {}
        self._tokens = []
        for token_id, row in enumerate(vocab):
            token = str(row[0])
            self._tokens.append(token)
            token_to_token_id[token] = token_id

            if len(row) > 2 and row[2] == "BYTE":
                self._bytes[int(row[0][3:5], 16)] = token_id
                continue

            suffix_to_score[token] = float(row[1])
            for i in range(1, len(token)):
                suffix_to_score[token[i:]] = suffix_to_score.get(token[i:], math.nan)

        for i in range(256):
            if self._bytes[i] == 0:
                raise ValueError(f"Byte token for <0x{i:02X}> is not set.")

        suffixes = list(suffix_to_score.keys())
        suffixes.append("")
        suffixes.sort(key=lambda x: x[::-1])

        suffix_to_id: dict[str, int] = {}
        num_pieces = 0
        for suffix in suffixes:
            suffix_to_id[suffix] = num_pieces
            if suffix != "":
                piece_code = (ord(suffix[0]) << 32) | suffix_to_id[suffix[1:]]
                self._to_suffix_id[piece_code] = np.int32(num_pieces)
            num_pieces += 1 + sum(suffix[:i] in suffix_to_score for i in range(1, len(suffix) + 1))

        self._table = np.zeros((num_pieces, 4), dtype=np.int32)
        row_idx = 0
        for suffix in suffixes:
            for piece_length in range(len(suffix), 0, -1):
                piece = suffix[:piece_length]
                score = suffix_to_score.get(piece, None)
                if score is None:
                    continue
                self._table[row_idx, TABLE_PIECE_LENGTH] = piece_length
                self._table[row_idx, TABLE_TOKEN_ID] = token_to_token_id.get(piece, -1)
                self._table[row_idx, TABLE_SCORE] = (
                    round(score * 1e4) if math.isfinite(score) else INVALID_SCORE
                )
                self._table[row_idx, TABLE_PIECE_ID] = suffix_to_id[piece]
                row_idx += 1

            self._table[row_idx, TABLE_PIECE_LENGTH] = 1
            self._table[row_idx, TABLE_TOKEN_ID] = -1
            self._table[row_idx, TABLE_SCORE] = UNKNOWN_SCORE
            row_idx += 1

    def encode(self, data: str) -> np.ndarray:
        data_points = np.frombuffer(data.encode("utf-32"), dtype=np.int32)[1:]
        scores = np.full((len(data_points) + 1,), 2**60, dtype=np.int64)
        scores[-1] = 0
        path = np.zeros((len(data_points) + 1, 3), dtype=np.int32)
        suffix_id = 0

        for i in range(len(data_points) - 1, -1, -1):
            c = int(data_points[i])
            for p in range(suffix_id, len(self._table)):
                piece_code = (c << 32) | int(self._table[p, TABLE_PIECE_ID])
                suffix_id = int(self._to_suffix_id.get(piece_code, 0))
                if suffix_id > 0 or self._table[p, TABLE_SCORE] == UNKNOWN_SCORE:
                    break

            for p in range(suffix_id, len(self._table)):
                score = int(self._table[p, TABLE_SCORE])
                if score > INVALID_SCORE:
                    piece_length = int(self._table[p, TABLE_PIECE_LENGTH])
                    s = scores[i + piece_length] - score
                    if s < scores[i]:
                        scores[i] = s
                        path[i, PATH_TOKEN_LENGTH] = piece_length
                        path[i, PATH_TOKEN_ID] = int(self._table[p, TABLE_TOKEN_ID])
                        path[i, PATH_NUM_TOKENS] = path[i + piece_length, PATH_NUM_TOKENS] + 1
                        if score == UNKNOWN_SCORE:
                            path[i, PATH_NUM_TOKENS] += (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)

                if score == UNKNOWN_SCORE:
                    break

        pos = 0
        token_ids = np.zeros(path[0, PATH_NUM_TOKENS], dtype=np.int32)
        token_pos = 0
        while pos < len(data_points):
            if path[pos, PATH_TOKEN_ID] >= 0:
                token_ids[token_pos] = path[pos, PATH_TOKEN_ID]
                token_pos += 1
            else:
                c = int(data_points[pos])
                n_bytes = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)
                for i in range(n_bytes):
                    if n_bytes == 1:
                        byte = c
                    elif i == 0:
                        byte = (0xF00 >> n_bytes) & 0xFF
                    else:
                        byte = 0x80
                    byte |= (c >> ((n_bytes - i - 1) * 6)) & 0x3F
                    token_ids[token_pos] = self._bytes[byte]
                    token_pos += 1

            if path[pos, PATH_TOKEN_LENGTH] <= 0:
                raise ValueError(f"Invalid token path at position {pos}")
            pos += path[pos, PATH_TOKEN_LENGTH]

        return token_ids

    def encode_as_tokens(self, data: str) -> list[str]:
        return [self._tokens[token_id] for token_id in self.encode(data)]


class Plamo2Tokenizer(PreTrainedTokenizer):  # type: ignore[misc]
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<|plamo:unk|>",
        bos_token: str = "<|plamo:bos|>",
        eos_token: str = "<|plamo:bos|>",
        pad_token: str = "<|plamo:pad|>",
        cls_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("add_bos_token", False)
        kwargs.setdefault("add_eos_token", False)
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        self.vocab = {v[0]: i for i, v in enumerate(self.data)}
        self.aho_corasick = AhoCorasick()
        self.aho_corasick.build(self.data)
        self.vocab_file = vocab_file
        self.add_bos_token = kwargs["add_bos_token"]
        self.add_eos_token = kwargs["add_eos_token"]

        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @classmethod
    def _get_additional_special_tokens(cls, tokenizer_config: dict[str, Any]) -> list[str]:
        tokens = tokenizer_config.get("additional_special_tokens")
        if tokens:
            return list(tokens)

        tokens = tokenizer_config.get("extra_special_tokens")
        if tokens:
            return list(tokens)

        core_special_tokens = {
            tokenizer_config.get("unk_token"),
            tokenizer_config.get("bos_token"),
            tokenizer_config.get("eos_token"),
            tokenizer_config.get("pad_token"),
            tokenizer_config.get("cls_token"),
            tokenizer_config.get("sep_token"),
            tokenizer_config.get("mask_token"),
        }
        extra_tokens = []
        for token_id in sorted(
            tokenizer_config.get("added_tokens_decoder", {}),
            key=int,
        ):
            token_info = tokenizer_config["added_tokens_decoder"][token_id]
            content = token_info.get("content")
            if (
                token_info.get("special")
                and content not in core_special_tokens
                and content not in extra_tokens
            ):
                extra_tokens.append(content)
        return extra_tokens

    @classmethod
    def from_pretrained(cls, model_path: str | Path):
        model_path = Path(model_path)
        tokenizer_config_path = model_path / "tokenizer_config.json"
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        return cls(
            vocab_file=str(model_path / VOCAB_FILES_NAMES["vocab_file"]),
            unk_token=tokenizer_config.get("unk_token", "<|plamo:unk|>"),
            bos_token=tokenizer_config.get("bos_token", "<|plamo:bos|>"),
            eos_token=tokenizer_config.get("eos_token", "<|plamo:bos|>"),
            pad_token=tokenizer_config.get("pad_token", "<|plamo:pad|>"),
            add_bos_token=tokenizer_config.get("add_bos_token", False),
            add_eos_token=tokenizer_config.get("add_eos_token", False),
            additional_special_tokens=cls._get_additional_special_tokens(
                tokenizer_config
            ),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["aho_corasick"] = None
        return state

    def __setstate__(self, d: dict[str, Any]) -> None:
        self.__dict__ = d
        self.aho_corasick = AhoCorasick()
        self.aho_corasick.build(self.data)

    @property
    def vocab_size(self):
        return len(self.data)

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens):
        return b"".join(
            [
                bytes([int(t[3:5], 16)]) if t.startswith("<0x") else t.encode("utf-8")
                for t in tokens
            ]
        ).decode("utf-8", errors="replace")

    def _tokenize(self, text: str):
        return self.aho_corasick.encode_as_tokens(text)

    def _convert_token_to_id(self, token: str):
        return self.vocab.get(token, 0)

    def _convert_id_to_token(self, index: int):
        return self.data[index][0]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []
        output = bos_token_id + token_ids_0 + eos_token_id
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id
        return output

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return ("",)

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "w", encoding="utf-8") as f:
                for token in self.data:
                    print(json.dumps(token, ensure_ascii=False), file=f)
        return (out_vocab_file,)

    def save_pretrained(
        self,
        save_directory: str | Path,
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        **kwargs: Any,
    ):
        self.init_kwargs["add_bos_token"] = self.add_bos_token
        self.init_kwargs["add_eos_token"] = self.add_eos_token
        files = super().save_pretrained(
            str(save_directory),
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            **kwargs,
        )

        tokenizer_config_name = (
            f"{filename_prefix}-tokenizer_config.json"
            if filename_prefix
            else "tokenizer_config.json"
        )
        tokenizer_config_path = Path(save_directory) / tokenizer_config_name
        if tokenizer_config_path.exists():
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)
            tokenizer_config["add_bos_token"] = self.add_bos_token
            tokenizer_config["add_eos_token"] = self.add_eos_token
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
                f.write("\n")

        return files


def _apply_ja01_template(dummy_image_token: str, text: str, image_length: int = 1) -> str:
    return "\n".join(
        [
            "以下はタスクを説明する指示で、文脈を説明した入力とペアになっています。要求を適切に補完するよう応答を書いてください。",
            "",
            "### 指示:",
            dummy_image_token * image_length + text,
            "",
            "### 応答:",
            "",
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int
) -> tuple[int, int]:
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor = min(aspect_ratio, target_aspect_ratio) / max(aspect_ratio, target_aspect_ratio)
        if factor > best_factor:
            best_factor = factor
            best_ratio = ratio
        elif factor == best_factor:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess_eagle2_image(
    image: PIL.Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 448, use_thumbnail: bool = False
) -> list[PIL.Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios_ = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios_, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _dynamic_preprocess_eagle2(
    text: str,
    images: Sequence[PIL.Image.Image],
    max_tiles: int,
    image_size: int,
    context_length: int,
    num_image_token: int,
    image_start_token: str,
    image_end_token: str,
    dummy_image_token: str,
) -> tuple[str, list[list[PIL.Image.Image]]]:
    text_parts = text.split(dummy_image_token)
    if len(text_parts) - 1 != len(images):
        raise ValueError(
            "The number of images and image tokens do not match: "
            f"{len(images)} vs {len(text_parts) - 1}"
        )
    new_text_buf = io.StringIO()
    max_input_tiles_limited_by_context = max_tiles
    per_tile_len = num_image_token
    while max_input_tiles_limited_by_context > 0:
        image_tiles = [
            _dynamic_preprocess_eagle2_image(
                image,
                image_size=image_size,
                max_num=max_input_tiles_limited_by_context,
                use_thumbnail=True,
            )
            for image in images
        ]
        if sum([2 + len(tiles) * per_tile_len for tiles in image_tiles]) < context_length:
            break
        max_input_tiles_limited_by_context -= 2 if max_input_tiles_limited_by_context >= 3 else 1

    for text_part, tiles in zip(text_parts[:-1], image_tiles):
        num_patches = len(tiles)
        image_tokens = image_start_token + dummy_image_token * num_image_token * num_patches + image_end_token
        new_text_buf.write(text_part)
        new_text_buf.write(image_tokens)
    new_text_buf.write(text_parts[-1])
    return new_text_buf.getvalue(), image_tiles


def _pad_image_to_multiple_of_stride(image: PIL.Image.Image, stride: int) -> PIL.Image.Image:
    width, height = image.size
    padded_width = math.ceil(width / stride) * stride
    padded_height = math.ceil(height / stride) * stride
    if (padded_width, padded_height) == (width, height):
        return image
    padded = PIL.Image.new("RGB", (padded_width, padded_height))
    padded.paste(image, (0, 0))
    return padded


class Plamo2VLProcessor:
    preserve_pad_token = True

    def __init__(
        self,
        image_processor: SiglipImageProcessor,
        tokenizer: Plamo2Tokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self._dummy_image_token = "<plamo:image>"
        self._image_size = image_processor.size["width"]
        self._patch_size = int(image_processor.patch_size)
        self._max_image_width = int(image_processor.max_image_width)
        self._max_image_height = int(image_processor.max_image_height)
        self._downsample_ratio = float(image_processor.downsample_ratio)
        self._tile_context_length = int(image_processor.tile_context_length)

    @classmethod
    def from_pretrained(cls, model_path: str | Path):
        model_path = Path(model_path)
        image_processor = SiglipImageProcessor.from_pretrained(model_path)
        tokenizer = Plamo2Tokenizer.from_pretrained(model_path)
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)

    def save_pretrained(self, save_directory: str | Path, **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        image_processor_files = self.image_processor.save_pretrained(
            str(save_directory), **kwargs
        )
        tokenizer_files = self.tokenizer.save_pretrained(str(save_directory), **kwargs)

        return tuple(dict.fromkeys([*image_processor_files, *tokenizer_files]))

    def format_prompt(self, text: str, image_length: int = 0) -> str:
        return _apply_ja01_template(self._dummy_image_token, text, image_length)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ):
        del add_generation_prompt, kwargs
        if isinstance(messages, list):
            if messages and isinstance(messages[0], str):
                prompt = "\n".join(messages)
            else:
                chunks = []
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    chunks.append(str(content))
                prompt = "\n".join(chunks)
        else:
            prompt = str(messages)
        if tokenize:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        return prompt

    def _split_images_into_tiles(
        self,
        text: str,
        images: list[PIL.Image.Image],
    ) -> tuple[str, list[PIL.Image.Image]]:
        image_size = self._image_size
        if self._downsample_ratio != 1.0:
            downsampled_stride = self._patch_size * int(1 / self._downsample_ratio)
            image_size = math.ceil(image_size / downsampled_stride) * downsampled_stride
        f_h = image_size // self._patch_size
        num_image_token = int((f_h * self._downsample_ratio) ** 2)
        new_text, image_tiles = _dynamic_preprocess_eagle2(
            text=text,
            images=images,
            max_tiles=24,
            image_size=image_size,
            context_length=self._tile_context_length,
            num_image_token=num_image_token,
            image_start_token="<siglip2:img>",
            image_end_token="</siglip2:img>",
            dummy_image_token=self._dummy_image_token,
        )
        images = [tile for tiles in image_tiles for tile in tiles]
        if self._downsample_ratio != 1.0:
            downsampled_stride = self._patch_size * int(1 / self._downsample_ratio)
            images = [_pad_image_to_multiple_of_stride(img, downsampled_stride) for img in images]
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        return new_text, images

    def process(self, text, images, padding: bool = False, return_tensors: Optional[str] = None, **kwargs):
        del return_tensors, kwargs
        if isinstance(text, list):
            if len(text) != 1:
                raise ValueError("Only batch size 1 is currently supported")
            text = text[0]
        if images is None:
            raise ValueError("images must be provided")
        if isinstance(images, PIL.Image.Image):
            images = [images]
        if len(images) == 0:
            raise ValueError("At least one image is required")

        formatted_text = self.format_prompt(str(text), image_length=len(images))
        tiled_text, tiled_images = self._split_images_into_tiles(formatted_text, list(images))
        text_inputs = self.tokenizer(
            [tiled_text],
            return_attention_mask=True,
            padding=padding,
        )
        image_inputs = self.image_processor(images=tiled_images, return_tensors="np")
        return {
            "input_ids": np.array(text_inputs["input_ids"], dtype=np.int64),
            "attention_mask": np.array(text_inputs["attention_mask"], dtype=np.int64),
            "pixel_values": np.array(image_inputs["pixel_values"]),
        }

    __call__ = process
