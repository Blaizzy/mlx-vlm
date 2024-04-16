# MLX-VLM

MLX-VLM a package for running Vision LLMs on your Mac using MLX.


## Get started

The easiest way to get started is to install the `mlx-vlm` package:

**With `pip`**:

```sh
pip install mlx-vlm
```

**Inference**:

```sh
python -m mlx_vlm.generate --model mlx-community/llava-1.5-7b-hf-4bit --max-tokens 10 --temp 0.0
```

**Convert models**:

```sh
pip -m mlx_vlm.convert
    --hf-path  llava-hf/llava-1.5-7b-hf\
    -q \
    --upload-repo mlx-community/llava-1.5-7b-hf-4bit
```