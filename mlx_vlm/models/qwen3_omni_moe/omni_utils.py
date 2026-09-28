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
        mask = model_inputs["feature_attention_mask"]
        lengths = mask.sum(axis=1)
        # feature_attention_mask is sample-domain, while the audio encoder counts
        # mel frames; convert each item's length via the mask/mel hop ratio so
        # audio_feature_lengths matches the encoder's true frame count. Otherwise
        # the sample-domain length (~160x too large) is forwarded to the model,
        # which skips the mel-frame conversion in Thinker.get_audio_features and
        # blows up the audio position/placeholder tensors (see #1620). This mirrors
        # the same conversion already done in the processor and thinker.
        input_features = model_inputs.get("input_features")
        if input_features is not None:
            mel_frames = input_features.shape[-1]
            mask_len = mask.shape[-1]
            if mask_len > mel_frames:
                lengths = lengths // (mask_len // mel_frames)
        model_inputs["audio_feature_lengths"] = lengths.astype(mx.int32)

    return model_inputs, text
