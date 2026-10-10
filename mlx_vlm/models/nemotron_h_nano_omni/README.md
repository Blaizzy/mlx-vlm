# Nemotron 3 Nano Omni

Nemotron 3 Nano Omni is a multimodal reasoning family from NVIDIA. The current implementation supports text, image, and audio inputs.

## Model

- [nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit](https://huggingface.co/nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit)

## Generate

```sh
mlx_vlm.generate \
  --model nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Use `--audio /path/to/audio.wav` for audio prompts. The checkpoint's pruned video path requires efficient video sampling, which is not implemented yet.
