import mlx.core as mx
import numpy as np

from ..idefics3 import LanguageModel
from ..idefics3 import Model as Idefics3Model
from ..idefics3 import ModelConfig, TextConfig, VisionConfig, VisionModel


class Model(Idefics3Model):
    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        # Assumes bs == 1

        B, T, D_text = inputs_embeds.shape
        N, S, D_img = image_features.shape

        image_offset = 0
        cur_embeds = inputs_embeds[0]

        # Find positions of <image> tokens in the text
        image_token_index = self.config.image_token_index
        image_positions = np.where(input_ids == image_token_index)[1].tolist()
        num_image_tokens = len(image_positions)

        # If no <image> => text-only
        if num_image_tokens == 0:
            empty_slice = image_features[0][:0, :]  # shape (0, D)
            return mx.concatenate([cur_embeds, empty_slice], axis=0)

        # Typically, if each image is S embeddings, we expect the total # of <image> tokens
        # in this sample to be multiple of S => each group of S tokens = 1 image
        if num_image_tokens % S != 0:
            raise ValueError(
                f"Input has {num_image_tokens} <image> tokens, not a multiple of S={S}. "
                "Cannot map them to blocks of shape (S, D)."
            )

        chunks = [image_positions[i : i + S] for i in range(0, num_image_tokens, S)]

        segments = []
        text_start = 0

        # For each chunk (each chunk => 1 image)
        for chunk in chunks:
            cur_block = image_features[image_offset]
            image_offset += 1

            # We'll iterate over the S positions in ascending order
            for i_s, pos in enumerate(chunk):
                if pos > text_start:
                    segments.append(cur_embeds[text_start:pos])
                # Then add one row from cur_block => shape (1, D)
                row_of_block = cur_block[i_s : i_s + 1, :]
                segments.append(row_of_block)
                text_start = pos + 1

        # leftover text after the final <image> token
        if text_start < T:
            segments.append(cur_embeds[text_start:])

        # cat them into a single (T_b, D) tensor
        merged_sample = mx.concatenate(segments, axis=0)
        return mx.expand_dims(merged_sample, axis=0)
