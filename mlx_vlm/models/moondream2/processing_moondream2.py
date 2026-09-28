"""Processor for Moondream2.

The current Moondream2 checkpoints share Moondream3's tokenizer and image
pipeline, so this reuses the Moondream3 processor. The one difference: the
checkpoint bundles a stale GPT-2 ``tokenizer.json`` that the model does not
use (its own code loads ``moondream/starmie-v1``), so we force that tokenizer
here rather than picking up the bundled one.
"""

from ..base import install_auto_processor_patch, load_chat_template
from ..moondream3.processing_moondream3 import TOKENIZER_REPO, Moondream3Processor


class Moondream2Processor(Moondream3Processor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO, trust_remote_code=True
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)
        return cls(tokenizer=tokenizer)


install_auto_processor_patch(["moondream1", "moondream2"], Moondream2Processor)
