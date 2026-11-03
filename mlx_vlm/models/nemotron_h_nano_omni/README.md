# Nemotron Omni

Nemotron Omni is NVIDIA's multimodal Mamba-2/attention/MoE family with a C-RADIO vision encoder. This implementation supports Nemotron 3 Nano Omni, Nemotron 3.5 Super VL, and Nemotron 3.5 Bolt. Bolt's Puzzle backbone uses different expert counts, intermediate widths, and routing settings in each MoE layer.

## Model

- [nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit](https://huggingface.co/nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit)
- [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
- [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)

The Super VL checkpoint loads directly, preserving the NVFP4 expert weights and their per-expert global scales. FP8 dense layers are decoded to BF16. The checkpoint is approximately 75 GB on disk; allow additional memory for loading, vision processing, and the context cache. A 128 GB Mac can run short-context requests.

## Generate

```sh
mlx_vlm.generate \
  --model nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

For Super VL:

```sh
mlx_vlm.generate \
  --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256 \
  --temperature 0.6 \
  --top-p 0.95
```

Omit `--image` for text requests. Use `--video /path/to/clip.mp4 --fps 2` for a single video. Frames are packed in pairs (the last frame is repeated when needed), and efficient video sampling prunes redundant tokens while retaining the first pair. Image and video inputs must be sent in separate requests. The default video sampling cap is 128 frames.

For very short clips, the checkpoint's default pruning rate of 0.7 can retain only the first frame pair. Add `--gen-kwargs '{"video_pruning_rate": 0}'` to keep all video tokens, or pass `video_pruning_rate=0` to `generate()` in Python.

Use `--audio /path/to/audio.wav` with Nano Omni; Super VL and Bolt have no audio encoder.

## Bolt conversion

Bolt's BF16 checkpoint is approximately 150 GB before inference caches and vision processing. On a 128 GB Mac, convert it to MLX 4-bit first:

```sh
mlx_vlm.convert \
  --hf-path nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-bf16 \
  --mlx-path ./Nemotron-3.5-Bolt-4bit \
  --quantize --q-bits 4 --q-group-size 64

mlx_vlm.generate \
  --model ./Nemotron-3.5-Bolt-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Text and video requests use the same options described above. Machines with sufficient memory can load the original BF16 checkpoint directly. The released Bolt checkpoint contains no MTP tensors, so it cannot be used as its own `--draft-model`.

The default 4-bit conversion is approximately 43.1 GB on disk, with the vision encoder and projector kept in BF16. Short text, image, and video validation requests used less than 47 GB of peak MLX memory on an M5 Max; longer contexts require more memory.

## Native MTP

Super VL's bundled MTP head supports speculative decoding for text, image, and single-video requests. Pass the original checkpoint as both the target and drafter:

```sh
mlx_vlm.generate \
  --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --draft-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --draft-block-size 3 \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --temperature 0 \
  --max-tokens 256
```

MTP processes the complete prompt in one prefill pass. For video MTP, add `--gen-kwargs '{"video_pruning_rate": 0}'` (or `video_pruning_rate=0` in Python) so prompt tokens remain aligned with the target hidden states. Pruned video requests support ordinary generation only.

The MTP kind is detected automatically. Only the MTP shard is read for the drafter; it shares the target's token embeddings and output head. NVFP4 expert weights retain their individual global scales. A block size of 3 drafts two tokens; larger values recursively reuse the same native prediction module. The target verifies every draft, and rejected tokens roll back both attention and Mamba state.

To save the drafter separately:

```sh
python -m mlx_vlm.split_mtp \
  --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --output ./nemotron-super-vl-mtp
```

Use that directory as `--draft-model` with either the original target or a matching MLX conversion. Ordinary generation does not load the optional MTP head.
