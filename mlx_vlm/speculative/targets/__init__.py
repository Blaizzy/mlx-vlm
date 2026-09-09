"""Bind speculative execution without changing the serving model."""


def bind_speculative_target(model):
    if getattr(model, "model_type", None) == "deepseek_v4":
        from .deepseek_v4 import DeepseekV4SpeculativeTarget

        if not isinstance(model, DeepseekV4SpeculativeTarget):
            return DeepseekV4SpeculativeTarget(model)
    if getattr(model, "model_type", None) in ("glm5_next", "glm5_next_text"):
        from .glm5_next import Glm5NextSpeculativeTarget

        if not isinstance(model, Glm5NextSpeculativeTarget):
            return Glm5NextSpeculativeTarget(model)
    return model
