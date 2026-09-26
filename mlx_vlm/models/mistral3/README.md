# Mistral 3

Mistral 3 is a family of vision-language models with a Pixtral vision encoder and a Mistral language backbone. The implementation supports text and image prompts.

## Model

- [nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit](https://huggingface.co/nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit)

## Generate

```sh
mlx_vlm.generate \
  --model nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```
