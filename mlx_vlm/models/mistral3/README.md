# Mistral 3

Mistral 3 is a family of vision-language models with a Pixtral vision encoder and a Mistral language backbone. The implementation supports text and image prompts.

## Model

- [nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit](https://huggingface.co/nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit)
- [mlx-community/Mistral-Small-4-119B-2603-4bit](https://huggingface.co/mlx-community/Mistral-Small-4-119B-2603-4bit): uses a `mistral4` mixture-of-experts language model (128 experts, 4 active per token). The 4-bit weights need about 70 GB of memory.

## Generate

```sh
mlx_vlm.generate \
  --model nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```
