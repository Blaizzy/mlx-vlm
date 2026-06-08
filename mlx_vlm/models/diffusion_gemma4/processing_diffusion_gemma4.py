"""
Text-only processor for DiffusionGemma4 checkpoints.

The upstream checkpoint advertises ``Gemma4Processor`` in processor metadata,
which pulls in the remote video processor and therefore torch/torchvision. This
model port is text-only for now, so the local processor intentionally loads only
the tokenizer and chat template.
"""

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from ..base import load_chat_template, to_mlx


class DiffusionGemma4Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, chat_template=None, **kwargs):
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        images=None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if images is not None or audio is not None or videos is not None:
            raise ValueError(
                "DiffusionGemma4 vision/audio/video inputs are not supported yet."
            )
        if text is None:
            raise ValueError("Provide `text` for DiffusionGemma4Processor.")

        kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text=text, **kwargs)
        return BatchFeature(data=to_mlx(text_inputs))

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("tokenize", False)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(self.tokenizer.model_input_names)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from pathlib import Path

        from transformers import AutoTokenizer

        kwargs.pop("trust_remote_code", None)
        kwargs.pop("use_fast", None)

        model_path = Path(pretrained_model_name_or_path)
        is_local = model_path.exists() and model_path.is_dir()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path) if is_local else pretrained_model_name_or_path,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        return cls(tokenizer=tokenizer, chat_template=tokenizer.chat_template)


__all__ = ["DiffusionGemma4Processor"]

from ..base import install_auto_processor_patch

install_auto_processor_patch(
    ["diffusion_gemma4", "diffusion_gemma"],
    DiffusionGemma4Processor,
)
