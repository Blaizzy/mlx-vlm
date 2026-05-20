import json
import warnings

import mlx.core as mx
import numpy as np

from ..models.base import to_mlx
from ..prompt_utils import MODEL_CONFIG, apply_chat_template

NATIVE_PREPROCESS_MODELS = set(MODEL_CONFIG.keys())


def build_completion_mask(input_ids, assistant_id, end_turn_id=None, user_id=None):
    """Build a per-token mask for assistant completion tokens."""
    if assistant_id is None:
        raise ValueError("assistant_id is required when train_on_completions is enabled")

    if isinstance(input_ids, mx.array):
        ids_np = np.array(input_ids)
    else:
        ids_np = np.array(input_ids)

    original_shape = ids_np.shape
    if ids_np.ndim == 1:
        ids_np = ids_np[None, :]

    batch_size, seq_length = ids_np.shape
    mask = np.zeros((batch_size, seq_length), dtype=np.int32)

    for row_idx in range(batch_size):
        in_assistant = False
        for col_idx, tid in enumerate(ids_np[row_idx]):
            if tid == assistant_id:
                in_assistant = True
            elif end_turn_id is not None and tid == end_turn_id:
                if in_assistant:
                    mask[row_idx, col_idx] = 1
                in_assistant = False
            elif user_id is not None and tid == user_id:
                in_assistant = False

            if in_assistant:
                mask[row_idx, col_idx] = 1

    if len(original_shape) == 1:
        mask = mask[0]

    return mx.array(mask)


def resolve_completion_token_ids(
    processor,
    assistant_id=None,
    end_turn_id=None,
    user_id=None,
    logger=None,
):
    """Resolve role and turn-boundary token ids from the processor chat template."""
    if assistant_id is not None and end_turn_id is not None and user_id is not None:
        return assistant_id, end_turn_id, user_id

    tokenizer = getattr(processor, "tokenizer", processor)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    decode = getattr(tokenizer, "decode", None)
    if apply_template is None or encode is None or decode is None:
        return assistant_id, end_turn_id, user_id

    try:
        probe_text = apply_template(
            [
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "A"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        probe_ids = encode(probe_text)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Auto-detection failed: {exc}")
        return assistant_id, end_turn_id, user_id

    role_keywords = {"model", "assistant", "bot"}
    user_keywords = {"user", "human"}
    end_turn_markers = {"<turn|>", "<|im_end|>", "<end_of_utterance>"}

    prev_is_special = False
    for tid in probe_ids:
        decoded = decode([tid])
        stripped = decoded.strip()
        cleaned = stripped.lower().strip("<>|")
        is_special = "<" in stripped and ">" in stripped
        role_position = is_special or prev_is_special

        if role_position and cleaned in role_keywords and assistant_id is None:
            assistant_id = tid
            if logger is not None:
                logger.info(f"Auto-detected assistant_id: {tid} ({repr(stripped)})")

        if role_position and cleaned in user_keywords and user_id is None:
            user_id = tid
            if logger is not None:
                logger.info(f"Auto-detected user_id: {tid} ({repr(stripped)})")

        if stripped in end_turn_markers and end_turn_id is None:
            end_turn_id = tid
            if logger is not None:
                logger.info(f"Auto-detected end_turn_id: {tid} ({repr(stripped)})")

        prev_is_special = is_special

    return assistant_id, end_turn_id, user_id


class VisionDataset:
    """Simplified dataset class for Vision LLMs"""

    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_resize_shape=None,
        train_on_completions=False,
        assistant_id=None,
        end_turn_id=None,
        user_id=None,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_resize_shape = image_resize_shape
        self.train_on_completions = train_on_completions
        self.assistant_id = assistant_id
        self.end_turn_id = end_turn_id
        self.user_id = user_id

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Streaming dataset has no length")

    def __getitem__(self, idx):
        return self.process(self.dataset[idx])

    def process(self, item):
        """Process a single item from the dataset"""
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        audio = item.get("audio", item.get("audios", []))
        if not isinstance(audio, list):
            audio = [audio] if audio else []

        conversations = item.get("messages", item.get("conversations"))

        model_type = self.config.get("model_type")

        num_images = len(images)
        num_audios = len(audio)

        prompts = []
        if isinstance(conversations, list) and isinstance(conversations[0], list):
            for conversation in conversations:
                if model_type == "pixtral":
                    conversation = [json.loads(i) for i in conversation]
                    if len(conversations) > 1:
                        warnings.warn(
                            "Pixtral batch processing is not supported yet. Set batch size to 1."
                        )

                prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    conversation,
                    add_generation_prompt=False,
                    num_images=num_images,
                    num_audios=num_audios,
                )
                prompts.append(prompt)

        else:
            if model_type == "pixtral":
                conversations = [json.loads(i) for i in conversations]
            prompt = apply_chat_template(
                self.processor,
                self.config,
                conversations,
                add_generation_prompt=False,
                num_images=num_images,
                num_audios=num_audios,
            )
            prompts.append(prompt)

        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        inputs = None
        if model_type in NATIVE_PREPROCESS_MODELS:
            try:
                inputs = process_inputs_with_fallback(
                    processor=self.processor,
                    prompts=prompts,
                    images=images if images else None,
                    audio=audio if audio else None,
                    add_special_tokens=False,
                )
                if "images" in inputs and "pixel_values" not in inputs:
                    inputs["pixel_values"] = inputs.pop("images")
            except Exception:
                pass

        if inputs is None:
            inputs = prepare_inputs(
                processor=self.processor,
                images=images if images else None,
                audio=audio if audio else None,
                prompts=prompts,
                image_token_index=image_token_index,
                resize_shape=self.image_resize_shape,
            )

        inputs = to_mlx(inputs)

        result = {
            "pixel_values": inputs.get("pixel_values"),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get(
                "attention_mask", mx.ones_like(inputs["input_ids"])
            ),
            **{
                k: v
                for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            },
        }

        if self.train_on_completions:
            result["completion_mask"] = build_completion_mask(
                result["input_ids"],
                self.assistant_id,
                self.end_turn_id,
                self.user_id,
            )

        return result


class PreferenceVisionDataset:
    """Dataset for preference-based training (ORPO, DPO).

    Expected item keys: ``chosen``, ``rejected``, and optionally ``images`` / ``image``.
    Each of ``chosen`` / ``rejected`` can be:
    - a list of message dicts (processed via ``apply_chat_template``)
    - a plain string (encoded directly)
    """

    def __init__(
        self,
        hf_dataset,
        config,
        processor,
        image_resize_shape=None,
        train_on_completions=False,
        assistant_id=None,
        end_turn_id=None,
        user_id=None,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.image_resize_shape = image_resize_shape
        self.train_on_completions = train_on_completions
        self.assistant_id = assistant_id
        self.end_turn_id = end_turn_id
        self.user_id = user_id

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Streaming dataset has no length")

    def __getitem__(self, idx):
        return self.process(self.dataset[idx])

    def process(self, item):
        from mlx_vlm.utils import prepare_inputs, process_inputs_with_fallback

        images = item.get("images", item.get("image", []))
        if not isinstance(images, list):
            images = [images] if images else []

        image_token_index = self.config.get("image_token_index") or self.config.get(
            "image_token_id"
        )
        if not image_token_index:
            raise ValueError(
                "Config must contain 'image_token_index' or 'image_token_id'"
            )

        num_images = len(images)

        model_type = self.config.get("model_type")

        result = {}
        for key in ("chosen", "rejected"):
            sequence = item[key]
            if isinstance(sequence, str):
                prompt = sequence
            else:
                prompt = apply_chat_template(
                    self.processor,
                    self.config,
                    sequence,
                    add_generation_prompt=False,
                    num_images=num_images,
                )
            inputs = None
            if model_type in NATIVE_PREPROCESS_MODELS:
                try:
                    inputs = process_inputs_with_fallback(
                        processor=self.processor,
                        prompts=[prompt],
                        images=images if images else None,
                        audio=None,
                        add_special_tokens=False,
                    )
                    if "images" in inputs and "pixel_values" not in inputs:
                        inputs["pixel_values"] = inputs.pop("images")
                except Exception:
                    pass

            if inputs is None:
                inputs = prepare_inputs(
                    processor=self.processor,
                    images=images if images else None,
                    audio=None,
                    prompts=[prompt],
                    image_token_index=image_token_index,
                    resize_shape=self.image_resize_shape,
                )

            inputs = to_mlx(inputs)

            result[f"{key}_input_ids"] = inputs["input_ids"]
            result[f"{key}_attention_mask"] = inputs.get(
                "attention_mask", mx.ones_like(inputs["input_ids"])
            )
            if self.train_on_completions:
                result[f"{key}_completion_mask"] = build_completion_mask(
                    inputs["input_ids"],
                    self.assistant_id,
                    self.end_turn_id,
                    self.user_id,
                )
            if inputs.get("pixel_values") is not None:
                result[f"{key}_pixel_values"] = inputs["pixel_values"]

        return result
