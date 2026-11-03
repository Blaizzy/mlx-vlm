"""Lazy VLM media preparation, fallback processing, and preference masks."""

import unittest
from unittest.mock import patch

import mlx.core as mx

from mlx_vlm.trainer.vlm.preference.dataset import PreferenceVisionDataset
from mlx_vlm.trainer.vlm.sft.dataset import VisionDataset


class VisionPreparationTest(unittest.TestCase):
    def helpers(self, calls):
        def template(processor, config, messages, *, add_generation_prompt, **kwargs):
            return "prefix" if add_generation_prompt else "full"

        def generic(**kwargs):
            calls.append(kwargs)
            return {
                "input_ids": [[1, 2] if kwargs["prompts"] == ["prefix"] else [1, 2, 3]],
                "images": mx.ones((1, 2)),
                "input_values": mx.ones((1, 2)),
                "video_values": mx.ones((1, 2)),
            }

        def native(**kwargs):
            raise RuntimeError("Processor-specific path unavailable")

        return {"native": {}}, template, generic, native

    @patch("mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers")
    def test_media_grouping_is_lazy_and_fallback_preserves_completion_masks(self, load):
        calls = []
        load.return_value = self.helpers(calls)
        row = {
            "messages": [
                {"role": "user", "content": "look"},
                {"role": "assistant", "content": "answer"},
            ],
            "picture": "custom.jpg",
            "images": "fallback.jpg",
            "audio": {"array": [0.1, 0.2]},
            "videos": "clip.mp4",
        }
        dataset = VisionDataset(
            [row],
            {"model_type": "native", "image_token_index": 7},
            object(),
            config={
                "image_feature": "picture",
                "train_on_completions": True,
                "image_resize_shape": (8, 8),
            },
        )
        self.assertEqual(dataset.media_signature(0), ("image", "audio", "video"))
        self.assertEqual(calls, [])
        prepared = dataset[0]
        self.assertEqual(prepared["input_ids"].tolist(), [1, 2, 3])
        self.assertEqual(prepared["completion_mask"].tolist(), [0, 0, 1])
        self.assertIn("pixel_values", prepared)
        self.assertNotIn("images", prepared)
        for kwargs in calls:
            self.assertEqual(kwargs["images"], ["custom.jpg"])
            self.assertEqual(kwargs["audio"], [[0.1, 0.2]])
            self.assertEqual(kwargs["videos"], ["clip.mp4"])
            self.assertEqual(kwargs["image_token_index"], 7)
            self.assertEqual(kwargs["resize_shape"], (8, 8))
            self.assertFalse(kwargs["add_special_tokens"])
        row["picture"] = []
        self.assertEqual(dataset.media_signature(0), ("audio", "video"))

    @patch("mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers")
    def test_flexible_rows_keep_template_fields_and_tools(self, load):
        rendered = []

        def template(processor, config, messages, **kwargs):
            rendered.append((messages, kwargs))
            return "full"

        def generic(**kwargs):
            return {"input_ids": [[1, 2, 3]]}

        load.return_value = ({}, template, generic, None)
        rows = [
            {
                "conversations": [
                    {"from": "human", "value": "hi"},
                    {"from": "gpt", "value": "yo", "reasoning_content": "think"},
                    {"role": "assistant", "tool_calls": [{"id": "1"}]},
                ],
                "tools": '[{"type": "function"}]',
            },
            {"prompt": "q", "completion": "a"},
            {"text": "<raw>"},
        ]
        dataset = VisionDataset(
            rows,
            {"model_type": "other"},
            object(),
            config={"chat_template_kwargs": {"enable_thinking": True}},
        )
        for i in range(3):
            dataset[i]

        messages, kwargs = rendered[0]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "assistant"])
        self.assertEqual(messages[0]["from"], "human")
        self.assertEqual(messages[1]["reasoning_content"], "think")
        self.assertEqual(messages[2]["content"], "")
        self.assertEqual(kwargs["tools"], [{"type": "function"}])
        self.assertTrue(kwargs["enable_thinking"])
        self.assertEqual([m["content"] for m in rendered[1][0]], ["q", "a"])
        self.assertEqual(len(rendered), 2)

    @patch("mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers")
    def test_preference_default_config_preserves_shared_media_and_response_masks(
        self, load
    ):
        calls = []
        load.return_value = self.helpers(calls)
        dataset = PreferenceVisionDataset(
            [
                {
                    "prompt": "look",
                    "chosen": "yes",
                    "rejected": "no",
                    "image": "shared.jpg",
                }
            ],
            {"model_type": "generic"},
            object(),
        )
        for candidate in dataset[0].values():
            self.assertEqual(candidate["completion_mask"].tolist(), [0, 0, 1])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call["images"] == ["shared.jpg"] for call in calls))


if __name__ == "__main__":
    unittest.main()
