# MLX-VLM

MLX-VLM a package for running Vision LLMs on your Mac using MLX.


## Get started

The easiest way to get started is to install the `mlx-vlm` package:

**With `pip`**:

```sh
pip install mlx-vlm
```

**Inference**:

**CLI**
```sh
python -m mlx_vlm.generate --model qnguyen3/nanoLLaVA --max-tokens 100 --temp 0.0
```

**Chat UI with Gradio**
```sh
python -m mlx_vlm.chat_ui --model qnguyen3/nanoLLaVA
```