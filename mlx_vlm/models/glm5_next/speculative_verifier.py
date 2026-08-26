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

    def __call__(
        self,
        language_model,
        inputs: mx.array,
        *,
        cache: Any = None,
        capture_layer_ids: Optional[List[int]] = None,
        skip_logits: bool = False,
    ) -> LanguageModelOutput:
        del capture_layer_ids  # glm5_next MTP uses the final hidden only
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


__all__ = ["Glm5NextExactSpeculativeVerifier", "verify_logits"]
