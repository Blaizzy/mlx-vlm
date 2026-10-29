import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import LanguageModelOutput
from ..minicpm_ocr.precision import preserve_norm_precision
from ..qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES
from ..qwen3_5.qwen3_5 import Model as QwenModel


class Model(QwenModel):
    def __init__(self, config):
        super().__init__(config)
        preserve_norm_precision(self.language_model)
        if hasattr(self.language_model, "lm_head"):
            del self.language_model.lm_head
        self.score = nn.Linear(
            config.text_config.hidden_size, config.num_labels, bias=False
        )

    def sanitize(self, weights):
        weights = {
            key: (
                value.astype(mx.float32)
                if key.endswith(NORM_WEIGHT_SUFFIXES)
                else value
            )
            for key, value in weights.items()
        }
        return super().sanitize(weights)

    @property
    def cast_predicate(self):
        predicate = super().cast_predicate
        return lambda path: predicate(path) and not path.endswith(NORM_WEIGHT_SUFFIXES)

    def __call__(self, input_ids, pixel_values=None, mask=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Jev requires nonempty batched input_ids.")
        mask = kwargs.pop("attention_mask", mask)
        valid = input_ids != self.config.pad_token_id
        if mx.any(mx.sum(valid, axis=-1) == 0):
            raise ValueError("Jev cannot score an all-padding sequence.")
        if mask is not None and mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids.")
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        if mask is not None and not mx.all(mask):
            # The shared Qwen recurrent prefill path derives padding from its
            # cache, which a sequence classifier does not keep. Remove padding
            # per row after encoding images so padded tokens cannot change the
            # recurrent state. Keep the already-computed multimodal positions.
            pooled = []
            for row, row_mask in enumerate(np.asarray(mask)):
                indices = np.flatnonzero(row_mask)
                if not len(indices) or not np.all(np.diff(indices) == 1):
                    raise ValueError(
                        "Jev expects contiguous tokens with left or right padding."
                    )
                start, end = int(indices[0]), int(indices[-1]) + 1
                ids = input_ids[row : row + 1, start:end]
                positions = features.position_ids
                positions = (
                    positions[row : row + 1, start:end]
                    if positions.ndim == 2
                    else positions[:, row : row + 1, start:end]
                )
                output = self.language_model(
                    ids,
                    inputs_embeds=features.inputs_embeds[row : row + 1, start:end],
                    position_ids=positions,
                    return_hidden=True,
                    skip_logits=True,
                )
                last = mx.argmax(
                    (ids[0] != self.config.pad_token_id) * mx.arange(end - start)
                )
                pooled.append(output.hidden_states[-1][0, last])
            return LanguageModelOutput(logits=self.score(mx.stack(pooled)))
        output = self.language_model(
            input_ids,
            mask=mask,
            return_hidden=True,
            skip_logits=True,
            **features.to_dict(),
        )
        # Match the reference sequence classifier for left and right padding.
        last = mx.argmax(
            valid.astype(mx.int32) * mx.arange(input_ids.shape[1]), axis=-1
        )
        hidden = output.hidden_states[-1][mx.arange(input_ids.shape[0]), last]
        return LanguageModelOutput(logits=self.score(hidden))

    @property
    def quant_predicate(self):
        predicate = super().quant_predicate
        return lambda path, module: path != "score" and (
            predicate is None or predicate(path, module)
        )
