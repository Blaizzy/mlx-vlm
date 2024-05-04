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
python -m mlx_vlm.generate --model qnguyen3/nanoLLaVA --max-tokens 100 --temp 0.0
```

### Example Calls

* Idefics 2 w/ Local Path

```sh
python -m mlx_vlm.generate --model mlx-community/idefics2-8b-4bit --max-tokens 1000 --temp 0.0  --prompt "Describe this image" --image /path/to/file.png
```

* Idefics 2 w/ Local Path

```sh
python -m mlx_vlm.generate --model mlx-community/idefics2-8b-4bit --max-tokens 1000 --temp 0.0  --prompt "Is this a logo?" --image https://pypi.org/static/images/logo-small.8998e9d1.svg
```

