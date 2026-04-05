# Contributing

To work on MLX-VLM in editable mode run:

```bash
pip install -e .
```

Check that the model weights are available in the `safetensors` format, convert if necessary and add the model file to `mlx_vlm/models`.

Tests can be run from the `mlx_vlm/` directory:

```bash
python -m unittest discover tests/
```

Please format code using `pre-commit` before submitting a pull request.

