# Phi-4 推理视觉 (SigLIP2)

Microsoft 的 Phi-4 多模态推理模型，结合了 Phi-3 语言主干与 SigLIP2 NaFlex 视觉编码器。支持具有强大推理能力的可变分辨率图像理解。

## 模型

| | |
|---|---|
| **模型 ID** | `microsoft/Phi-4-reasoning-vision-15B` |
| **架构** | Phi-3 (语言) + SigLIP2 NaFlex (视觉) + MLP 2x GELU 投影器 |
| **参数** | ~15B |
| **视觉编码器** | SigLIP2 与 NaFlex（可变分辨率，256-3600 个 patch）|
| **任务** | 视觉问答、图像推理、图像描述 |

## CLI 使用

```bash
python -m mlx_vlm.generate \
    --model microsoft/Phi-4-reasoning-vision-15B \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

使用自定义生成参数：

```bash
python -m mlx_vlm.generate \
    --model microsoft/Phi-4-reasoning-vision-15B \
    --image path/to/image.jpg \
    --prompt "What objects are in this image?" \
    --max-tokens 512 \
    --temp 0.0
```

## Python 使用

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_image_processor

model, processor = load("microsoft/Phi-4-reasoning-vision-15B")

image = "path/to/image.jpg"
prompt = "Describe this image."

formatted = apply_chat_template(processor, model.config, prompt, num_images=1)

output = generate(
    model,
    processor,
    formatted,
    [image],
    max_tokens=512,
    temperature=0.0,
)
print(output)
```

## 架构

- **视觉**：SigLIP2 与 NaFlex 动态 patching。图像根据分辨率处理为可变数量的 patch（256-3600），实现高效处理不同的纵横比。
- **投影器**：两层 MLP，具有GELU 激活，将视觉特征（1152 维）映射到语言模型的隐藏空间（5120 维）。
- **语言**：Phi-3 解码器，具有 40 层，分组查询注意力（40 个头，10 个 KV 头），融合 QKV/gate-up 投影和 RoPE。

## 说明

- 模型使用 `<image>` 作为图像占位符 token，带有 `IMAGE_TOKEN_INDEX = -200`。
- NaFlex 每个图像产生可变长度的图像特征序列，因此有效上下文长度取决于输入图像分辨率。
- 视觉编码器使用双三次插值进行动态位置嵌入调整大小。
