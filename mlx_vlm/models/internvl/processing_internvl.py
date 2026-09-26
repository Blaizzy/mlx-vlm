from ..base import install_auto_processor_patch
from ..internvl_chat.processor import IMG_CONTEXT_TOKEN, InternVLChatProcessor


class InternVLProcessor(InternVLChatProcessor):
    def __call__(self, text=None, images=None, **kwargs):
        if isinstance(text, str):
            text = [text]
        if images is None:
            return self.tokenizer(text, **kwargs)
        text = [prompt.replace(IMG_CONTEXT_TOKEN, "<image>") for prompt in text]
        inputs = super().__call__(text=text, images=images, **kwargs)
        inputs.pop("num_patches_list", None)
        return inputs


install_auto_processor_patch("internvl", InternVLProcessor)
