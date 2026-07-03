def install_transformers_legacy_registration_shims():
    """Allow legacy string-based Auto* registrations on newer transformers."""
    try:
        from transformers import AutoProcessor, AutoTokenizer
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
        from transformers.models.auto.tokenization_auto import (
            REGISTERED_FAST_ALIASES,
            REGISTERED_TOKENIZER_CLASSES,
        )
    except Exception:
        return

    if not getattr(AutoTokenizer.register, "_mlx_vlm_legacy_shim", False):
        original_register_tokenizer = AutoTokenizer.register

        def register_tokenizer_compat(
            config_class,
            tokenizer_class=None,
            slow_tokenizer_class=None,
            fast_tokenizer_class=None,
            exist_ok=False,
        ):
            if isinstance(config_class, str):
                if tokenizer_class is None:
                    if fast_tokenizer_class is not None:
                        tokenizer_class = fast_tokenizer_class
                    elif slow_tokenizer_class is not None:
                        tokenizer_class = slow_tokenizer_class
                    else:
                        raise ValueError("You need to pass a `tokenizer_class`")

                for candidate in (
                    slow_tokenizer_class,
                    fast_tokenizer_class,
                    tokenizer_class,
                ):
                    if candidate is not None:
                        REGISTERED_TOKENIZER_CLASSES[candidate.__name__] = candidate

                if (
                    slow_tokenizer_class is not None
                    and fast_tokenizer_class is not None
                ):
                    REGISTERED_FAST_ALIASES[slow_tokenizer_class.__name__] = (
                        fast_tokenizer_class
                    )
                return

            return original_register_tokenizer(
                config_class,
                tokenizer_class=tokenizer_class,
                slow_tokenizer_class=slow_tokenizer_class,
                fast_tokenizer_class=fast_tokenizer_class,
                exist_ok=exist_ok,
            )

        register_tokenizer_compat._mlx_vlm_legacy_shim = True
        AutoTokenizer.register = staticmethod(register_tokenizer_compat)

    if not getattr(AutoProcessor.register, "_mlx_vlm_legacy_shim", False):
        original_register_processor = AutoProcessor.register

        def register_processor_compat(
            config_class, processor_class, exist_ok=False
        ):
            if isinstance(config_class, str):
                for registered_processor in PROCESSOR_MAPPING._extra_content.values():
                    if registered_processor is processor_class:
                        return
                PROCESSOR_MAPPING._extra_content[processor_class] = processor_class
                return

            return original_register_processor(
                config_class, processor_class, exist_ok=exist_ok
            )

        register_processor_compat._mlx_vlm_legacy_shim = True
        AutoProcessor.register = staticmethod(register_processor_compat)
