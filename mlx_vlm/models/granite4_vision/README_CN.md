# Granite 4.0 Vision

IBM 的 Granite 4.0 Vision 模型，具有 DeepStack 多层特征注入和 WindowQFormer 投影器。结合了 SigLIP 视觉编码器与 GraniteMoeHybrid 语言模型。

## 模型

| | |
|---|---|
| **模型 ID** | `ibm-granite/granite-4.0-3b-vision` |
| **架构** | GraniteMoeHybrid LM + SigLIP (视觉) + WindowQFormer 投影器 + DeepStack |
| **参数** | ~3B |
| **视觉编码器** | SigLIP (27 层, 1152 hidden, 384px, patch 16) |
| **投影器** | 4 个 DeepStack + 4 个空间 WindowQFormerDownsampler 模块 |
| **任务** | 文档理解、VQA、图像描述、表格/图表分析 |

## CLI 使用

```bash
python -m mlx_vlm.generate \
    --model ibm-granite/granite-4.0-3b-vision \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."
```

## Python 使用

```python
from mlx_vlm import load, generate

model, processor = load("ibm-granite/granite-4.0-3b-vision")

image = "path/to/image.jpg"
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image."}
    ]}
]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

output = generate(model, processor, image=image, prompt=prompt, max_tokens=512)
print(output)
```

## 架构

- **视觉**：SigLIP 编码器（27 层, patch 大小 16）
- **语言**：GraniteMoeHybrid，具有 MUP 乘数和融合门+up MLP
- **DeepStack**：来自层 [-19, -13, -7, -1] 的视觉特征在 LLM 层 [9, 6, 3, 0] 处注入
- **空间采样**：来自最后视觉层的 4 个偏移组（TL/TR/BL/BR）→ LLM 层 [12, 15, 18, 21]
- **WindowQFormer**：窗口式交叉注意力，带有面积插值下采样（4/8 速率）
- **LoRA**：加载期间自动合并适配器权重（r=256, alpha=256）
