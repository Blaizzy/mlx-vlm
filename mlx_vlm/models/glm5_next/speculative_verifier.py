from typing import Any, List, Optional

import mlx.core as mx

from ..base import LanguageModelOutput
from ..exact_speculative_verify import exact_speculative_verify_weight


def verify_logits(language_model, normed_hidden: mx.array) -> mx.array:
    """(shared) LM head for a speculative-verify block.

    Uses the exact-speculative-verify dense matmul when the head is dense BF16/FP16 so
    a block verify's argmax matches singleton decode; quantized heads fall back to the
    ordinary projection (still a valid target distribution).
    """
    if language_model.args.tie_word_embeddings:
        embed = language_model.model.embed_tokens
        out = exact_speculative_verify_weight(embed.weight, normed_hidden)
        return out if out is not None else embed.as_linear(normed_hidden)
    head = language_model.lm_head
    out = exact_speculative_verify_weight(head.weight, normed_hidden)
    if out is None:
        return head(normed_hidden)
    if "bias" in head:
        out = out + head.bias
    return out


class Glm5NextExactSpeculativeVerifier:
    """Speculative-verify forward for glm5_next.

    Runs the model over a proposed block and returns, alongside the (exact) verify
    logits, the pre-final-norm hidden the nextn drafter consumes and each KDA layer's
    per-step recurrent state used by ``rollback_speculative_cache``.
    """

    @staticmethod
    def requires_tokenwise(language_model) -> bool:
        for layer in language_model.model.layers:
            projection = getattr(layer.self_attn, "q_proj", None)
            if projection is not None:
                mode = getattr(projection, "mode", None)
                return mode == "mxfp8" or (
                    mode == "affine" and getattr(projection, "bits", None) == 8
                )
        return False

    def __call__(
        self,
        language_model,
        inputs: mx.array,
        *,
        cache: Any = None,
        capture_layer_ids: Optional[List[int]] = None,
        skip_logits: bool = False,
    ) -> LanguageModelOutput:
        if self.requires_tokenwise(language_model) and inputs.shape[1] > 1:
            return self._tokenwise(
                language_model,
                inputs,
                cache=cache,
                capture_layer_ids=capture_layer_ids,
                skip_logits=skip_logits,
            )
        gdn_sink: List = []
        hidden_sink: List = []
        normed = language_model.model(
            inputs, cache=cache, gdn_sink=gdn_sink, hidden_sink=hidden_sink
        )
        logits = None if skip_logits else verify_logits(language_model, normed)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={},
        )

    def _tokenwise(
        self,
        language_model,
        inputs: mx.array,
        *,
        cache: Any,
        capture_layer_ids: Optional[List[int]],
        skip_logits: bool,
    ) -> LanguageModelOutput:
        del capture_layer_ids
        gdn_steps = []
        hidden_steps = []
        normed_steps = []
        for position in range(inputs.shape[1]):
            gdn_sink: List = []
            hidden_sink: List = []
            normed_steps.append(
                language_model.model(
                    inputs[:, position : position + 1],
                    cache=cache,
                    gdn_sink=gdn_sink,
                    hidden_sink=hidden_sink,
                )
            )
            gdn_steps.append(gdn_sink)
            hidden_steps.append(hidden_sink)
        gdn_sink = self._merge_gdn_steps(gdn_steps)
        hidden_sink = [
            mx.concatenate([step[index] for step in hidden_steps], axis=1)
            for index in range(len(hidden_steps[0]))
        ]
        normed = mx.concatenate(normed_steps, axis=1)
        logits = None if skip_logits else verify_logits(language_model, normed)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=gdn_sink,
            shared_kv_states={},
        )

    @staticmethod
    def _merge_gdn_steps(gdn_steps: List[List]) -> List:
        if not gdn_steps:
            return []
        merged = []
        for states in zip(*gdn_steps):
            first = states[0]
            sequence = [
                mx.concatenate([state[index] for state in states], axis=1)
                for index in range(5)
            ]
            conv_input = mx.concatenate(
                [
                    first[8][:, : first[9] - 1],
                    *[state[8][:, -1:] for state in states],
                ],
                axis=1,
            )
            merged.append(
                (
                    *sequence,
                    first[5],
                    first[6],
                    first[7],
                    conv_input,
                    first[9],
                    first[10],
                )
            )
        return merged


__all__ = ["Glm5NextExactSpeculativeVerifier", "verify_logits"]
