# Falcon-OCR

Falcon-OCR 是 TII 的一个 300M 参数早期融合视觉语言模型，针对文档 OCR 进行了优化。它支持多种提取模式，并与布局检测器集成用于结构化文档处理。

它是一个单一的 Transformer，从第一层开始在共享参数空间中处理图像补丁和文本 token，使用混合注意力掩码，其中图像 token 进行双向关注，文本 token 基于图像进行因果解码。

## 模型

- **模型 ID**：`tiiuae/Falcon-OCR`
- **参数**：300M
- **支持的类别**：`plain`、`text`、`table`、`formula`、`caption`、`footnote`、`list-item`、`page-footer`、`page-header`、`section-header`、`title`

### 链接

- [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception) -- 代码和推理引擎
- [tiiuae/Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) -- HuggingFace 模型卡

## 何时使用什么

| 模式 | 最适合 | 使用方法 |
|------|----------|-----|
| **Plain OCR** | 简单文档、真实照片、幻灯片、收据、截图 | `generate(model, processor, "plain", image=...)` |
| **Layout + OCR** | 复杂的多列文档、学术论文、报告、密集页面（如报纸） | `generate_with_layout(model, processor, image=...)` |

## 安装

```bash
pip install mlx-vlm
```

Layout+OCR 还需要 PyTorch（用于布局检测器）：

```bash
pip install torch
```

## Python 示例

### Plain OCR
默认情况下，类别为 `"plain"`（通用文本提取）。您可以指定一个类别以使用特定于任务的提示。
```python
from mlx_vlm import load, generate

model, processor = load("tiiuae/Falcon-OCR")

output = generate(model, processor, "plain", image="document.png", max_tokens=2000)
print(output.text)

output = generate(model, processor, "formula", image="equation.png", max_tokens=512)
print(output.text)

output = generate(model, processor, "table", image="table.png", max_tokens=512)
print(output.text)
```


### Layout + OCR（密集文档）

使用 [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) 检测文档区域（文本、表格、公式、图形等），然后按阅读顺序对每个文本区域运行 OCR。

```python
from mlx_vlm import load

from mlx_vlm.models.falcon_ocr import generate_with_layout

model, processor = load("tiiuae/Falcon-OCR")

regions = generate_with_layout(model, processor, image="paper.png", max_tokens=4096)

for region in regions:
    print(f"[{region['category']}] {region.get('text', '')}")
```

输出中的每个区域包含：
- `category` -- 检测到的区域类型（文本、表格、公式、图形等）
- `bbox` -- 原始图像像素中的边界框坐标 `[x1, y1, x2, y2]`
- `score` -- 检测置信度
- `text` -- 该区域的提取文本
