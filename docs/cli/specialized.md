# Specialized CLIs

Beyond the primary commands ([`generate`](generate.md), [`server`](server.md), [`convert`](convert.md), [`chat`](chat.md), [`lora`](lora.md)), MLX-VLM ships a set of advanced, model-specific, and evaluation entry points invoked as Python modules. Each is a directory rather than a full flag reference: for the authoritative, version-current options, run any command with `--help`.

## Evaluation

Multimodal benchmark harnesses, run as `python -m mlx_vlm.evals.<name>`.

| Command | Purpose |
|---------|---------|
| `python -m mlx_vlm.evals.mmmu` | Evaluate a model on MMMU (Massive Multi-discipline Multimodal Understanding). |
| `python -m mlx_vlm.evals.mmstar` | Evaluate a model on the MMStar benchmark. |
| `python -m mlx_vlm.evals.ocrbench` | Evaluate a model on the OCRBench benchmark. |
| `python -m mlx_vlm.evals.math_vista` | Evaluate a model on the MathVista benchmark. |

## Model-specific scripts

Per-model conversion and inference scripts, run as `python -m mlx_vlm.models.<model>.<script>`.

| Command | Purpose |
|---------|---------|
| `python -m mlx_vlm.models.deepseek_v4.convert` | Convert a DeepSeek-V4 Flash Vision mixed checkpoint to MLX. |
| `python -m mlx_vlm.models.ernie_image.convert` | Convert an ERNIE-Image Diffusers checkpoint to MLX. |
| `python -m mlx_vlm.models.mage_flow.convert` | Convert a Mage-Flow Diffusers checkpoint to MLX. |
| `python -m mlx_vlm.models.ming_image.convert` | Convert Ming-Image-0.1-Design to MLX. |
| `python -m mlx_vlm.models.minimax_h3.convert` | Convert an official MiniMax-H3 workflow to MLX. |
| `python -m mlx_vlm.models.moge3.convert` | Convert an official MoGe-3 checkpoint to an MLX model repo. |
| `python -m mlx_vlm.models.nemotron_voicechat.convert` | Prepare an upload-ready mlx-vlm artifact from Nemotron VoiceChat safetensors. |
| `python -m mlx_vlm.models.pp_doclayout_v3.convert` | Convert PP-DocLayoutV3 to MLX. |
| `python -m mlx_vlm.models.qwen_image.convert` | Convert Qwen-Image-2.1 to MLX. |
| `python -m mlx_vlm.models.rfdetr.convert` | Convert RF-DETR detection weights to MLX. |
| `python -m mlx_vlm.models.rt_detr_v2.convert` | Convert HuggingFace RT-DETRv2 checkpoints to MLX safetensors. |
| `python -m mlx_vlm.models.sam3_1.convert_weights` | Convert a SAM 3.1 Meta checkpoint to MLX safetensors. |
| `python -m mlx_vlm.models.sam3d_body.convert_weights` | Convert SAM 3D Body weights to MLX safetensors. |
| `python -m mlx_vlm.models.sam3d_objects.convert` | Convert released SAM 3D Objects inference checkpoints to MLX. |
| `python -m mlx_vlm.models.yolo11.convert` | Convert the OmniParser icon_detect (YOLO11) checkpoint to MLX safetensors. |
| `python -m mlx_vlm.models.z_image.convert` | Convert a Z-Image Diffusers checkpoint to MLX. |
| `python -m mlx_vlm.models.rfdetr.generate` | Run RF-DETR object detection / segmentation on an image. |
| `python -m mlx_vlm.models.sam3.generate` | Run SAM3 detection, segmentation, and video tracking. |
| `python -m mlx_vlm.models.sam3_1.generate` | Run SAM 3.1 detection / segmentation inference. |
| `python -m mlx_vlm.models.sam3d_body.generate` | Run SAM 3D Body pose / mesh prediction on an image. |
| `python -m mlx_vlm.models.sam3d_objects.generate` | Run SAM 3D Objects image-to-3D (single request or JSONL stream). |
| `python -m mlx_vlm.models.sam3d_body.video` | Run SAM 3D Body prediction over video frames. |

## Speculative-decoding tools

Extract native multi-token-prediction (MTP) tensors into standalone MLX drafter models.

| Command | Purpose |
|---------|---------|
| `python -m mlx_vlm.split_mtp` | Extract a model's native MTP tensors into a standalone MLX drafter (generic). |
| `python -m mlx_vlm.speculative.drafters.deepseek_v4_mtp.split` | Split DeepSeek-V4 native MTP tensors into a standalone MLX drafter. |
| `python -m mlx_vlm.speculative.drafters.deepseek_v4_dspark.split` | Split DeepSeek-V4 DSpark MTP tensors into a standalone MLX drafter. |
| `python -m mlx_vlm.speculative.drafters.glm_moe_dsa_mtp.split` | Split GLM-MoE-DSA native MTP tensors into an MLX drafter. |
| `python -m mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split` | Split GLM-4.7-Flash native MTP tensors into a standalone MLX drafter. |
| `python -m mlx_vlm.speculative.drafters.glm5_next_mtp.split` | Split GLM-5-Next native MTP tensors into an MLX drafter. |
| `python -m mlx_vlm.speculative.drafters.inkling_mtp.split` | Split Inkling native MTP tensors into a standalone MLX drafter. |
| `python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split` | Split Qwen3.5 native MTP tensors into a standalone MLX drafter. |

## Utilities

| Command | Purpose |
|---------|---------|
| `python -m mlx_vlm.token_classification` | Tag and redact entities with an MLX token classifier (windowed inference + span decoding). |
