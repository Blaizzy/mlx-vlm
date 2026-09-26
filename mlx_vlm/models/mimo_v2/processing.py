from pathlib import Path

import mlx.core as mx

from ..base import install_auto_processor_patch
from ..qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor


class MiMoV2Processor(Qwen2_5_VLProcessor):
    supports_multiple_audio = True

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        audio_tokenizer_path = kwargs.pop("audio_tokenizer_path", None)
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.audio_token = "<|audio_pad|>"
        self.audio_tokenizer_path = audio_tokenizer_path
        self._audio_tokenizer = None

    @property
    def audio_tokenizer(self):
        if self._audio_tokenizer is None:
            from mlx.utils import tree_flatten
            from mlx_audio.codec.models.mimo_audio_tokenizer import MiMoAudioTokenizer
            from mlx_audio.codec.models.mimo_audio_tokenizer.config import ModelConfig
            from mlx_audio.utils import load_config, load_weights

            config = ModelConfig.from_dict(load_config(self.audio_tokenizer_path))
            model = MiMoAudioTokenizer(config)
            weights = model.sanitize(load_weights(self.audio_tokenizer_path))
            expected = {key for key, _ in tree_flatten(model.parameters())}
            encoder_expected = {key for key in expected if key.startswith("encoder.")}
            if not encoder_expected.issubset(weights):
                missing = sorted(encoder_expected - weights)
                raise ValueError(
                    f"Audio tokenizer encoder is missing weights: {missing}"
                )
            weights = {
                key: (
                    value.astype(mx.float32)
                    if ".quantizer." in key
                    else value.astype(mx.bfloat16)
                )
                for key, value in weights.items()
                if key in expected
            }
            model.load_weights(list(weights.items()), strict=False)
            model.eval()
            mx.eval(model.encoder.parameters())
            self._audio_tokenizer = model
        return self._audio_tokenizer

    def __call__(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        audio_codes = None
        if audio is not None and len(audio) > 0:
            if not isinstance(audio, list):
                audio = [audio]
            codes = [
                self.audio_tokenizer.encode(
                    item,
                    sample_rate=16000,
                    num_quantizers=20,
                )
                for item in audio
            ]
            index = 0
            for row in range(len(text)):
                parts = text[row].split(self.audio_token)
                expanded = parts[0]
                for part in parts[1:]:
                    if index >= len(codes):
                        raise ValueError(
                            "Prompt has more audio tokens than audio inputs"
                        )
                    count = (codes[index].shape[1] + 3) // 4
                    expanded += self.audio_token * count + part
                    index += 1
                text[row] = expanded
            if index != len(codes):
                raise ValueError("Audio inputs do not match prompt audio tokens")
            audio_codes = mx.concatenate(codes, axis=1).T
            # The server runs the processor on the request thread and the model
            # on the generation thread; a lazy graph built on this thread's
            # stream fails there with "There is no Stream(gpu, N) in current
            # thread". Materialise the codes where they are built.
            mx.eval(audio_codes)
        result = super().__call__(
            images=images,
            text=text,
            videos=videos,
            **kwargs,
        )
        if audio_codes is not None:
            result["audio_codes"] = audio_codes
            result["audio_code_lengths"] = [code.shape[1] for code in codes]
        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor.audio_tokenizer_path = (
            Path(pretrained_model_name_or_path) / "audio_tokenizer"
        )
        return processor


install_auto_processor_patch("mimo_v2", MiMoV2Processor)


__all__ = ["MiMoV2Processor"]
