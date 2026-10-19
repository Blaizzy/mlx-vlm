"""Prompt-only VLM preparation, retaining answers solely for reward callbacks."""

from collections.abc import Mapping

import mlx.core as mx

from mlx_vlm.trainer.vlm.sft.dataset import (
    VisionDataset,
    _as_media_list,
    _get_option,
    _media_value,
    _resolve_media,
)


class GRPOVisionDataset(VisionDataset):
    """Prepare chat messages or preformatted prompts for GRPO rollouts.

    Set ``apply_chat_template=False`` in dataset_config to tokenize an already
    formatted prompt string as supplied, without adding roles or a generation
    prefix. Such rows contain the complete instruction in ``prompt``.
    """

    def process(self, row):
        if not isinstance(row, Mapping):
            raise TypeError("GRPO rows must be mappings.")
        prompt_key = _get_option(self.args, "prompt_feature", "prompt")
        answer_key = _get_option(self.args, "answer_feature", "answer")
        prompt = row.get(prompt_key)
        apply_template = _get_option(self.args, "apply_chat_template", True)
        if not apply_template:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "Preformatted GRPO rows require a non-empty prompt string."
                )
            if row.get("system"):
                raise ValueError(
                    "Include the system instruction in the preformatted prompt, not a separate system field."
                )
        elif isinstance(prompt, str):
            conversation = [{"role": "user", "content": prompt}]
            if row.get("system"):
                conversation.insert(0, {"role": "system", "content": row["system"]})
        else:
            conversation = self._conversation(
                dict(row, messages=prompt) if isinstance(prompt, list) else row
            )
        if apply_template and conversation[-1].get("role") == "assistant":
            raise ValueError("GRPO messages must end with a prompt, not an answer.")
        images = [
            _resolve_media(item, self.base_path, image=True)
            for item in _as_media_list(
                _media_value(row, self.image_feature, ("images", "image"))
            )
        ]
        for name, key, alternatives in self._media_fields[1:]:
            if _as_media_list(_media_value(row, key, alternatives)):
                raise ValueError(
                    f"GRPO currently supports text and images, not {name}."
                )
        formatted = (
            self._apply_chat_template(
                self.processor,
                self.config,
                conversation,
                add_generation_prompt=True,
                num_images=len(images),
                num_audios=0,
            )
            if apply_template
            else prompt
        )
        inputs = self._prepare(formatted, images, [], [])
        # Remove processor padding before rollout; never truncate image slots.
        positions = [
            i for i, present in enumerate(inputs["attention_mask"].tolist()) if present
        ]
        if not positions:
            raise ValueError("GRPO requires a non-empty prompt.")
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("Prompt attention_mask must select contiguous tokens.")
        inputs["input_ids"] = inputs["input_ids"][positions[0] : positions[-1] + 1]
        inputs["attention_mask"] = mx.ones_like(inputs["input_ids"])
        return {
            "inputs": inputs,
            "prompt": prompt if prompt is not None else conversation,
            "answer": row.get(answer_key),
            "metadata": dict(row),
        }

    def __init__(self, dataset, model_config, processor, config=None):
        super().__init__(dataset, model_config, processor, config=config)
        self.args = config
        self.train_on_completions = False
