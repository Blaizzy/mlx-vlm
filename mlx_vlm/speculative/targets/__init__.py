"""Bind speculative execution without changing the serving model."""


def bind_speculative_target(model):
    if getattr(model, "model_type", None) in ("glm5_next", "glm5_next_text"):
        from .glm5_next import Glm5NextSpeculativeTarget

        if not isinstance(model, Glm5NextSpeculativeTarget):
            return Glm5NextSpeculativeTarget(model)
    return model
