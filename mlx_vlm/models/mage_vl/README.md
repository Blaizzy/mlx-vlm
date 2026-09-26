# Mage-VL

Mage-VL is a codec-native vision-language model from Microsoft supporting text, image, and video inputs. It combines a Mage-ViT visual encoder with a Qwen3 language backbone.

## Model

- [nativ-community/Mage-VL-OptiQ-4bit](https://huggingface.co/nativ-community/Mage-VL-OptiQ-4bit)

## Generate

Image:

```sh
mlx_vlm.generate \
  --model nativ-community/Mage-VL-OptiQ-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Video:

```sh
mlx_vlm.generate \
  --model nativ-community/Mage-VL-OptiQ-4bit \
  --video /path/to/video.mp4 \
  --prompt "Summarize this video." \
  --max-tokens 256
```
