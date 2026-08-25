from collections.abc import Mapping


def _config_value(config, name):
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _target_metadata(target_model):
    language_model = getattr(target_model, "language_model", target_model)
    outer_config = getattr(language_model, "config", None)
    target_config = _config_value(outer_config, "text_config") or outer_config
    if target_config is None:
        target_config = getattr(language_model, "args", None)
    inner = getattr(language_model, "model", language_model)
    layers = getattr(inner, "layers", None)
    return language_model, target_config, layers


def validate_dflash_target(config, target_model, algorithm: str) -> None:
    language_model, target_config, layers = _target_metadata(target_model)

    hidden_size = _config_value(target_config, "hidden_size")
    if hidden_size != config.hidden_size:
        raise ValueError(
            f"{algorithm} target hidden-size mismatch: "
            f"draft={config.hidden_size}, target={hidden_size}."
        )
    layer_count = (
        len(layers)
        if layers is not None
        else _config_value(target_config, "num_hidden_layers")
    )
    if layer_count != config.num_target_layers:
        raise ValueError(
            f"{algorithm} target layer-count mismatch: "
            f"draft={config.num_target_layers}, target={layer_count}."
        )
    vocab_size = _config_value(target_config, "vocab_size")
    if vocab_size != config.vocab_size:
        raise ValueError(
            f"{algorithm} target vocabulary mismatch: "
            f"draft={config.vocab_size}, target={vocab_size}."
        )
    if not hasattr(language_model, "rollback_speculative_cache"):
        raise ValueError(
            f"{algorithm} target {type(language_model).__name__} does not expose "
            "speculative cache rollback support."
        )


__all__ = ["validate_dflash_target"]
