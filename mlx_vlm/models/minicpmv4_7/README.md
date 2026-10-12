# MiniCPM-V 4.7

MLX support for `MiniCPM-V-4.7-1B`. The model combines a SigLIP
vision tower with a Qwen3.5 hybrid language model. Canvas M-RoPE preserves
spatial relationships between thumbnails and detail crops, and the order of
video frames.

The custom processor uses NumPy and Pillow and does not require PyTorch,
Torchvision, or remote code. It supports text, images, multiple images, video,
image slicing, and `4x`/`16x` visual downsampling. It preserves the checkpoint's
chat template, including thinking and Qwen3-Coder tool-call formatting.

## Local checkpoints

Extract the supplied archive before loading it. The source checkpoint
can be loaded directly.

```bash
tar -xf ~/Downloads/MiniCPM-V-4.7-1B.tar -C ~/Downloads

mlx_vlm.generate \
  --model ~/Downloads/MiniCPM-V-4.7-1B \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 200 --temperature 0
```

To save the target in MLX format:

```bash
mlx_vlm.convert \
  --hf-path ~/Downloads/MiniCPM-V-4.7-1B \
  --mlx-path ~/Downloads/MiniCPM-V-4.7-1B-mlx-bf16 \
  --dtype bfloat16
```

## Video and finer image detail

```bash
mlx_vlm.generate \
  --model ~/Downloads/MiniCPM-V-4.7-1B \
  --video path/to/video.mp4 --fps 1 \
  --prompt "Describe what happens." \
  --max-tokens 200 --temperature 0 \
  --processor-kwargs '{"max_num_frames": 32, "max_slice_nums": 1}'
```

For finer image detail, pass
`--processor-kwargs '{"downsample_mode": "4x"}'`. The custom processor forwards
the selected mode to both the vision tower and canvas position calculation.
The default is `16x`.

Visual prompts currently use cold prefill when automatic prefix caching cannot
recover the crop grids needed for canvas positions. Normal KV caching and
chunked prefill are supported.

Reference architecture:
[Transformers MiniCPM-V 4.7](https://huggingface.co/docs/transformers/model_doc/minicpmv4_7).
