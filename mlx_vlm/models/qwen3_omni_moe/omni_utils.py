import mlx.core as mx
import numpy as np

from mlx_vlm.utils import load_audio


def process_multimodal_info(conversation, use_audio_in_video=False):
    audios = []
    images = []
    videos = []
    for msg in conversation:
        if "content" in msg:
            if isinstance(msg["content"], str):
                continue
            for part in msg["content"]:
                if part["type"] == "audio":
                    audios.append(part["audio"])
                elif part["type"] == "image":
                    images.append(part["image"])
                elif part["type"] == "video":
                    videos.append(part["video"])
    return audios, images, videos


def prepare_omni_inputs(
    processor,
    conversation,
    use_audio_in_video=False,
):
    audios, images, videos = process_multimodal_info(conversation, use_audio_in_video)

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    loaded_audios = []
    if audios:
        sr = processor.feature_extractor.sampling_rate
        for audio_path in audios:
            loaded_audios.append(load_audio(audio_path, sr=sr))

    inputs = processor(
        text=[text],
        audio=loaded_audios if loaded_audios else None,
        images=images if images else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    model_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, "numpy"):
            model_inputs[k] = mx.array(v.numpy())
        elif isinstance(v, np.ndarray):
            model_inputs[k] = mx.array(v)
        else:
            model_inputs[k] = v

    if (
        "feature_attention_mask" in model_inputs
        and "audio_feature_lengths" not in model_inputs
    ):
        model_inputs["audio_feature_lengths"] = (
            model_inputs["feature_attention_mask"].sum(axis=1).astype(mx.int32)
        )

    return model_inputs, text
