# MiniCPM-V 4.7

MLX support for `MiniCPM-V-4.7-1B` and `MiniCPM-V-4.7-35B-A3B`. The models combine
a SigLIP vision tower with a Qwen3.5 hybrid language model. The 1B model uses a
dense decoder; the 35B-A3B model uses 256 experts with 8 selected per token,
plus a shared expert. Canvas M-RoPE preserves
spatial relationships between thumbnails and detail crops, and the order of
video frames.

The custom processor uses NumPy and Pillow and does not require PyTorch,
Torchvision, or remote code. It supports text, images, multiple images, video,
image slicing, and `4x`/`16x` visual downsampling. It preserves the checkpoint's
chat template, including thinking and Qwen3-Coder tool-call formatting.

## Local checkpoints

Extract the supplied archive before loading it. Both source checkpoints
can be loaded directly; substitute `35B-A3B` for `1B` in the paths below.

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

The 35B-A3B BF16 checkpoint needs about 71 GB of memory for short multimodal
requests. To create a smaller local copy:

```bash
tar -xf ~/Downloads/MiniCPM-V-4.7-35B-A3B.tar -C ~/Downloads

mlx_vlm.convert \
  --hf-path ~/Downloads/MiniCPM-V-4.7-35B-A3B \
  --mlx-path ~/Downloads/MiniCPM-V-4.7-35B-A3B-mlx-4bit \
  -q --q-bits 4 --q-group-size 64 --dtype bfloat16

mlx_vlm.generate \
  --model ~/Downloads/MiniCPM-V-4.7-35B-A3B-mlx-4bit \
  --image path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 200 --temperature 0
```

The default quantization policy keeps the vision modules, embeddings, output
head, and linear attention at full precision, and uses 8-bit expert routers.
The resulting 4-bit checkpoint is about 23.5 GB; memory usage also depends on
image resolution and context length.

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
