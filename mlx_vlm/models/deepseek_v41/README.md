# DeepSeek V4.1 Flash

> WIP — model support scaffold. Implementation and benchmarks forthcoming.

## Source

- Upstream weights: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- Conversion target: [pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit](https://huggingface.co/pipenetwork/DeepSeek-V4.1-Flash-MLX-mixed-4_8bit)

## Planned scope

- `config.py` (`model_type: deepseek_v41`)
- Language model (Causal Encoder-Decoder + CSA2 + Engram + DSpark)
- Vision tower (DeepSeek-ViT + projector) and processor
- Weight conversion mapping
- Tests in `tests/test_models.py`
