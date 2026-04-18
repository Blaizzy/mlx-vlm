# DeepSeek-OCR

DeepSeek-OCR 是一个强大的 OCR 模型，基于 SAM + Qwen2 编码器架构，针对文档理解、文本提取和视觉定位任务进行了优化。

## 模型架构

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                      Dynamic Resolution                          │
                    ├─────────────────────────────────────────────────────────────────┤
Local Patches       │  768×768 → SAM (12×12×896) → Qwen2 (144×896) → Proj (144×1024) │
(1-6 patches)       │                                                                  │
                    ├─────────────────────────────────────────────────────────────────┤
Global View         │ 1024×1024 → SAM (16×16×896) → Qwen2 (256×896) → Proj (256×1024)│
(1 image)           │                                                                  │
                    └─────────────────────────────────────────────────────────────────┘
                                                    ↓
                              [local_patches, global_view, view_separator]
                                                    ↓
                                          Language Model (DeepSeek-V2)
```

## 提示格式

### 文档转 Markdown
将文档图像转换为结构化的 markdown 格式：
```
<image>
<|grounding|>Convert the document to markdown.
```

### 通用 OCR
从图像中提取所有文本：
```
<image>
<|grounding|>OCR this image.
```

### 自由 OCR（无布局）
提取文本但不保留布局结构：
```
<image>
Free OCR.
```

### 解析图表
提取和描述文档中的图表/图形：
```
<image>
Parse the figure.
```

### 图像描述
获取图像的详细描述：
```
<image>
Describe this image in detail.
```

### 文本定位（Grounding）
定位图像中的特定文本并获取边界框坐标：
```
<image>
Locate <|ref|>your text here<|/ref|> in the image.
```

输出格式：`<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`

坐标归一化到 0-1000 范围。

## CLI 示例

### 文档转 Markdown
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max-tokens 2000
```

### 通用 OCR
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image receipt.jpg \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000
```

### 自由 OCR
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image text_image.png \
    --prompt "Free OCR." \
    --max-tokens 500
```

### 文本定位
```bash
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image table.jpeg \
    --prompt "Locate <|ref|>Total assets<|/ref|> in the image." \
    --max-tokens 100
```

## Python 脚本示例

### 基本 OCR
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# 加载模型
model, processor = load("mlx-community/DeepSeek-OCR-bf16")

# OCR 提示
prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# 生成
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    temperature=0.0,
)
print(result.text)
```

### 文档转 Markdown
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

prompt = "<|grounding|>Convert the document to markdown."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model=model,
    processor=processor,
    image="paper.pdf",  # 也支持 PDF 页面
    prompt=formatted_prompt,
    max_tokens=2000,
    temperature=0.0,
)
print(result.text)
```

### 文本定位
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

# 定位特定文本
text_to_find = "Total liabilities"
prompt = f"Locate <|ref|>{text_to_find}<|/ref|> in the image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
(
    model=model,
    processor=processor,
    image="financial_table.png",
    prompt=formatted_prompt,
    max_tokens=100,
    temperature=0.0,
)

# 从输出中解析边界框
# 输出格式： <|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
import re
match = re.search(r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]', result.text)
if match:
    x1, y1, x2, y2 = map(int, match.groups())
    # 坐标归一化到 0-1000
    print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
```

### 批处理
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from pathlib import Path

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# 处理多个图像
image_dir = Path("documents/")
for image_path in image_dir.glob("*.png"):
    result = generate(
        model=model,
        processor=processor,
        image=str(image_path),
        prompt=formatted_prompt,
        max_tokens=1000,
    )
    print(f"\n--- {image_path.name} ---")
    print(result.text)
```

## 动态分辨率

DeepSeek-OCR 使用动态分辨率来高效处理不同大小的图像：

**默认配置：**
- **全局视图**：1×1024×1024 → 256 个视觉 token
- **局部补丁**：(1-6)×768×768 → (1-6)×144 个视觉 token 每个
- **视图分隔符**：1 个 token

**工作原理：**
1. 分析图像的宽高比以确定最佳补丁网格
2. 局部补丁（768×768）根据网格布局捕获细粒度细节
3. 全局视图（1024×1024）捕获整体上下文
4. 特征拼接：[local_patches, global_view, view_separator]

**Token 计算：**
- 每个局部补丁：144 个 token（通过 query_768 从 12×12 SAM 特征）
- 全局视图：256 个 token（通过 query_1024 从 16×16 SAM 特征）
- 视图分隔符：1 个 token
- **总计：(num_patches × 144) + 256 + 1 个视觉 token**

**按宽高比的示例补丁数量：**
| 图像大小 | 宽高比 | 网格 | 补丁数 | 总 Token 数 |
|------------|--------|------|---------|--------------|
| 800×600 | 4:3 | 3×2 | 6 | 1121 |
| 600×800 | 3:4 | 2×3 | 6 | 1121 |
| 1200×400 | 3:1 | 3×1 | 3 | 689 |
| 400×1200 | 1:3 | 1×3 | 3 | 689 |
| 1000×1000 | 1:1 | 1×1 | 1 | 401 |

### 控制动态分辨率

您可以通过 `cropping`、`min_patches` 和 `max_patches` 参数控制补丁数量：

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/DeepSeek-OCR-bf16")

prompt = "<|grounding|>OCR this image."
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

# 默认：动态分辨率，1-6 个补丁
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    # cropping=True, min_patches=1, max_patches=6 (默认值)
)

# 仅全局视图（更快，257 个 token）
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    cropping=False,
)

# 限制补丁数以平衡速度与细节
result = generate(
    model=model,
    processor=processor,
    image="document.png",
    prompt=formatted_prompt,
    max_tokens=1000,
    cropping=True,
    min_patches=1,
    max_patches=3,
)
```

**按配置的 Token 数量：**

| 配置 | 补丁数 | Token 数 |
|---------------|---------|--------|
| `cropping=False` | 0 | 257 |
| `max_patches=1` | 1 | 401 |
| `max_patches=3` | 1-3 | 401-689 |
| `max_patches=6` (默认) | 1-6 | 401-1121 |

### 动态分辨率的 CLI 示例

```bash
# 默认：动态分辨率，1-6 个补丁
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000

# 仅全局视图（更快，257 个 token）
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000 \
    --processor-kwargs '{"cropping": false}'

# 限制最多 3 个补丁
mlx_vlm.generate \
    --model mlx-community/DeepSeek-OCR-bf16 \
    --image document.png \
    --prompt "<|grounding|>OCR this image." \
    --max-tokens 1000 \
    --processor-kwargs '{"cropping": true, "max_patches": 3}'
```

## 特殊 Token

| Token | 描述 |
|-------|-------------|
| `<image>` | 提示中的图像占位符 |
| `<\|grounding\|>` | 启用定位/结构化输出模式 |
| `<\|ref\|>...<\|/ref\|>` | 标记要在图像中定位的文本 |
| `<\|det\|>...<\|/det\|>` | 边界框输出格式 |
| `<\|User\|>` | 用户轮次标记 |
| `<\|Assistant\|>` | 助手轮次标记 |

## 提示

1. **为了获得最佳 OCR 结果**，使用 `<|grounding|>` 前缀启用结构化输出模式
2. **对于表格**，模型会自动输出 HTML 表格格式
3. **温度 0.0** 建议用于 OCR 任务以获得确定性结果
4. **增加 max_tokens** 用于包含大量文本的文档（整页需要 2000+）
5. **定位坐标**归一化到 0-1000 范围；缩放到您的图像尺寸

## 限制

- 定位（`<|ref|>...<|/ref|>`）对于某些查询可能返回全图像坐标
- 最适合文档风格图像（表单、表格、收据、论文）
- 与结构化提示相比，自由式问题可能产生不太可靠的输出
- 所有查询都应以句号结尾以获得更好的性能
