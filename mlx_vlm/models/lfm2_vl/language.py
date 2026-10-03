from ..lfm2.language import LanguageModel as Lfm2LanguageModel


class LanguageModel(Lfm2LanguageModel):
    """LFM2-VL language tower.

    Reuses the text LFM2 ``LanguageModel`` so speculative decoding support
    (``capture_layer_ids`` hidden-state taps, ``speculative_verify`` exact
    verification, and conv-state cache rollback) works with the VL target.
    Only checkpoint key handling differs: VL weights carry the
    ``language_model.`` prefix.
    """

    def sanitize(self, weights):
        if self.config.tie_word_embeddings:
            weights.pop("language_model.lm_head.weight", None)

        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name:
                if param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)

            sanitized_weights[name] = param
        return sanitized_weights
