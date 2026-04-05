# 贡献

要在可编辑模式下处理 MLX-VLM，请运行：

```bash
pip install -e .
```

确保模型权重以 `safetensors` 格式可用，如有需要请转换并将模型文件添加到 `mlx_vlm/models`。

可在 `mlx_vlm/` 目录中运行测试：

```bash
python -m unittest discover tests/
```

在提交拉取请求之前，请使用 `pre-commit` 格式化代码。

