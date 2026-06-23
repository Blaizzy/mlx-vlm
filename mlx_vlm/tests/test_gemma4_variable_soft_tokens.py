import unittest

import mlx.core as mx
import numpy as np

from mlx_vlm.models.gemma4.config import VisionConfig
from mlx_vlm.models.gemma4.gemma4 import masked_scatter
from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor
from mlx_vlm.models.gemma4.vision import VisionModel


class TestGemma4VariableSoftTokens(unittest.TestCase):
    class _ImageProcessor:
        model_input_names = ["pixel_values"]

        def __init__(self):
            self.seen_max_soft_tokens = None

        def fetch_images(self, images):
            return images

        def __call__(self, images, **kwargs):
            self.seen_max_soft_tokens = kwargs.get("max_soft_tokens")
            return {"pixel_values": np.zeros((1, 3, 4, 4), dtype=np.float32)}, [3]

    class _Tokenizer:
        model_input_names = ["input_ids", "attention_mask"]
        image_token = "<image>"
        image_token_id = 7
        boi_token = "<boi>"
        eoi_token = "<eoi>"
        audio_token = "<audio>"
        audio_token_id = 8
        boa_token = "<boa>"
        eoa_token = "<eoa>"
        video_token = "<video>"
        video_token_id = 9
        chat_template = "mock"

        @property
        def init_kwargs(self):
            return {}

        def convert_tokens_to_ids(self, token):
            return {
                self.image_token: self.image_token_id,
                self.audio_token: self.audio_token_id,
                self.video_token: self.video_token_id,
            }.get(token, 0)

        def add_special_tokens(self, tokens):
            return None

        def __call__(self, text, **kwargs):
            texts = [text] if isinstance(text, str) else text
            input_ids = []
            for item in texts:
                ids = []
                i = 0
                while i < len(item):
                    if item.startswith(self.image_token, i):
                        ids.append(self.image_token_id)
                        i += len(self.image_token)
                    else:
                        ids.append(1)
                        i += 1
                input_ids.append(ids)
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(ids) for ids in input_ids],
            }

    def test_processor_accepts_vision_soft_tokens_alias(self):
        image_processor = self._ImageProcessor()
        tokenizer = self._Tokenizer()
        processor = Gemma4Processor.__new__(Gemma4Processor)
        processor.image_processor = image_processor
        processor.tokenizer = tokenizer
        processor.video_processor = None
        processor.feature_extractor = None
        processor.image_token = tokenizer.image_token
        processor.image_token_id = tokenizer.image_token_id
        processor.boi_token = tokenizer.boi_token
        processor.eoi_token = tokenizer.eoi_token
        processor.audio_token = tokenizer.audio_token
        processor.audio_token_id = tokenizer.audio_token_id
        processor.boa_token = tokenizer.boa_token
        processor.eoa_token = tokenizer.eoa_token
        processor.video_token = tokenizer.video_token
        processor.video_token_id = tokenizer.video_token_id
        processor.full_image_sequence = (
            f"{processor.boi_token}{processor.image_token * 280}{processor.eoi_token}"
        )
        processor.full_audio_sequence = None

        result = processor(
            images=["image.png"],
            text="<image> transcribe",
            vision_soft_tokens_per_image=1120,
        )

        self.assertEqual(image_processor.seen_max_soft_tokens, 1120)
        self.assertEqual(
            result["input_ids"][0].tolist().count(self._Tokenizer.image_token_id), 3
        )

    def test_processor_rejects_conflicting_soft_token_aliases(self):
        image_processor = self._ImageProcessor()
        tokenizer = self._Tokenizer()
        processor = Gemma4Processor.__new__(Gemma4Processor)
        processor.image_processor = image_processor
        processor.tokenizer = tokenizer
        processor.video_processor = None
        processor.feature_extractor = None
        processor.image_token = tokenizer.image_token
        processor.image_token_id = tokenizer.image_token_id
        processor.boi_token = tokenizer.boi_token
        processor.eoi_token = tokenizer.eoi_token
        processor.audio_token = tokenizer.audio_token
        processor.audio_token_id = tokenizer.audio_token_id
        processor.boa_token = tokenizer.boa_token
        processor.eoa_token = tokenizer.eoa_token
        processor.video_token = tokenizer.video_token
        processor.video_token_id = tokenizer.video_token_id
        processor.full_image_sequence = (
            f"{processor.boi_token}{processor.image_token * 280}{processor.eoi_token}"
        )
        processor.full_audio_sequence = None

        with self.assertRaisesRegex(ValueError, "must match"):
            processor(
                images=["image.png"],
                text="<image> transcribe",
                max_soft_tokens=280,
                vision_soft_tokens_per_image=1120,
            )

    def test_vision_tower_accepts_more_patches_than_default_output_length(self):
        config = VisionConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=0,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            patch_size=2,
            pooling_kernel_size=2,
            default_output_length=2,
            position_embedding_size=64,
        )
        model = VisionModel(config)
        pixel_values = mx.ones((1, 3, 8, 8))

        features = model(pixel_values)

        self.assertEqual(features.shape, (1, 4, 4))

    def test_masked_scatter_rejects_mismatched_feature_count(self):
        inputs = mx.zeros((1, 3, 2))
        mask = mx.array([[[False, False], [True, True], [True, True]]])
        features = mx.ones((1, 1, 2))

        with self.assertRaisesRegex(ValueError, "features and tokens do not match"):
            masked_scatter(inputs, mask, features)

    def test_masked_scatter_can_truncate_extra_features_when_requested(self):
        inputs = mx.zeros((1, 3, 2))
        mask = mx.array([[[False, False], [True, True], [False, False]]])
        features = mx.array([[[1.0, 2.0], [3.0, 4.0]]])

        output = masked_scatter(inputs, mask, features, allow_truncate=True)

        self.assertEqual(output.tolist(), [[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]]])


if __name__ == "__main__":
    unittest.main()
