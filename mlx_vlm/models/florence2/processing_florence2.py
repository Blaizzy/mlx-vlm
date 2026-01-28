from transformers.models.florence2.processing_florence2 import Florence2Processor

# Store the original __init__
_original_init = Florence2Processor.__init__


def _patched_init(self, image_processor=None, tokenizer=None, **kwargs):
    """Patched __init__ that adds image_token attributes to tokenizer if missing."""
    if tokenizer is not None:
        # Ensure tokenizer has image_token attribute
        if not hasattr(tokenizer, "image_token"):
            tokenizer.image_token = "<image>"

        # Ensure tokenizer has image_token_id attribute
        if not hasattr(tokenizer, "image_token_id"):
            vocab = tokenizer.get_vocab()
            if tokenizer.image_token in vocab:
                tokenizer.image_token_id = vocab[tokenizer.image_token]
            else:
                tokenizer.add_tokens([tokenizer.image_token], special_tokens=True)
                tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(
                    tokenizer.image_token
                )

    # Call original __init__
    _original_init(self, image_processor=image_processor, tokenizer=tokenizer, **kwargs)


# Apply the patch
Florence2Processor.__init__ = _patched_init
