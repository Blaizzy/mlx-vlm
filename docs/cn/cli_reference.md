# CLI 参考

MLX-VLM 提供了几个命令行入口点：

- `mlx_vlm.convert` – 将 Hugging Face 模型转换为 MLX 格式。
- `mlx_vlm.generate` – 对图像运行推理。
- `mlx_vlm.video_generate` – 从视频文件生成内容。
- `mlx_vlm.smolvlm_video_generate` – 轻量级视频生成。
- `mlx_vlm.chat_ui` – 启动交互式 Gradio 界面。
- `mlx_vlm.server` – 运行 FastAPI 服务器。

每个命令都接受 `--help` 以获取完整的使用信息。

