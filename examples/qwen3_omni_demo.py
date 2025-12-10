import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_vlm import load
from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

# Configuration
model_path = "mlx_qwen3_omni_4bit_path"
audio_path = "dev/question.wav"
image_path = "dev/cars.jpg"
output_audio_path = "output_mlx.wav"
speaker = "Ethan"  # chelsie / aiden

model, processor = load(model_path, trust_remote_code=True)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            # {"type": "image", "image": image_path},  # generate with image input will take some time
            {"type": "text", "text": "What do you hear?"},
        ],
    },
]

model_inputs, text = prepare_omni_inputs(processor, conversation)

# Ensure all inputs are on default device
for k, v in model_inputs.items():
    if isinstance(v, mx.array):
        model_inputs[k] = v

# Prepare kwargs for generate
generate_kwargs = {
    "input_ids": model_inputs["input_ids"],
    "pixel_values": model_inputs.get("pixel_values", None),
    "pixel_values_videos": model_inputs.get("pixel_values_videos", None),
    "image_grid_thw": model_inputs.get("image_grid_thw", None),
    "video_grid_thw": model_inputs.get("video_grid_thw", None),
    "input_features": model_inputs.get("input_features", None),
    "feature_attention_mask": model_inputs.get("feature_attention_mask", None),
    "audio_feature_lengths": model_inputs.get("audio_feature_lengths", None),
    "thinker_max_new_tokens": 512,
    "talker_max_new_tokens": 2048,
    "talker_temperature": 0.9,
    "return_audio": True,
    "speaker": speaker,
}

generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

thinker_result, audio_wav = model.generate(**generate_kwargs)

output_text = processor.decode(thinker_result.sequences[0].tolist())
print("\nOutput Text:", output_text)

if audio_wav is not None:
    print(f"Saving audio to {output_audio_path}...")
    if hasattr(audio_wav, "reshape"):
        audio_data = np.array(audio_wav.reshape(-1))
    else:
        audio_data = np.array(audio_wav)

    sf.write(
        output_audio_path,
        audio_data,
        samplerate=24000,
    )
    print("Audio saved.")
else:
    print("No audio generated.")
