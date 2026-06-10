---
name: model-integration
description: Use this skill when adding a new MLX-VLM model family, porting a Hugging Face multimodal model, fixing unsupported model_type errors, debugging weight loading, processor/chat-template issues, model remapping, sanitization, or model-specific generation behavior.
---

# Model Integration

Use this workflow for adding or debugging MLX-VLM model families under `mlx_vlm/models/`.

## First Pass

1. Reproduce the failure with the smallest local model path or Hugging Face repo that shows it.
2. Inspect `config.json` for `model_type`, nested `text_config` / `vision_config` / `audio_config`, quantization fields, and processor requirements.
3. Find the closest existing implementation in `mlx_vlm/models/` before writing a new pattern.
4. Check `mlx_vlm/utils.py` for model-type remapping and dynamic import behavior.

## Expected Model Package Shape

Most model families keep a folder under `mlx_vlm/models/<model_type>/` with a small set of focused modules:

- `__init__.py` exports the model classes and config classes needed by `utils.py`.
- `config.py` maps Hugging Face config dictionaries to model, text, vision, and audio configs.
- Model implementation files define MLX modules for the language, vision, audio, projector, or wrapper components.
- Optional processor files handle model-specific image, video, audio, or prompt formatting behavior.
- A model-specific `README.md` is useful when prompts, modalities, or limitations differ from the generic CLI.

Mirror an existing nearby family instead of inventing a new folder layout.

## Weight Loading Rules

- Treat safetensors keys as the contract. Print key names and shapes before changing model code.
- Add `sanitize` methods when source checkpoints need key renames, QKV splitting, MoE expert stacking, Conv2d layout transforms, or dropped non-MLX buffers.
- Keep weight conversion and sharding separate from runtime key sanitization unless the existing model family already combines them.
- Preserve quantization metadata. Check `quantization`, `quantization_config`, compressed-tensors formats, and per-layer predicates before changing weight load behavior.
- For Conv2d weights, verify whether the source layout needs transposition for MLX.

## Processor And Prompting

- Use the Hugging Face processor/chat template when it matches the model. Avoid hand-written prompt formats until you have checked the processor behavior.
- Verify image token counts, video frame handling, audio sampling assumptions, and any model-specific special tokens.
- For multimodal models, test text-only and one media input separately before testing multi-image, audio+image, or video flows.

## Verification Gates

- Compile/import the touched modules: `python -m compileall -q mlx_vlm/models/<family>`.
- Run a load-only smoke test if weights are available.
- Run one minimal generation test for the supported modality.
- Compare against Hugging Face or an existing reference when porting: tensor shapes first, then logits or generated output.
- Add a focused regression test when the change touches shared loading, remapping, processor behavior, or server-visible output.

Do not broaden shared utilities unless at least two model families need the behavior or the existing local pattern already points there.
