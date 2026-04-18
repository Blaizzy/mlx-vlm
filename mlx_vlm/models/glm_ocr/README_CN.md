# GLM-OCR

GLM-OCR 是一个专为光学字符识别（OCR）任务优化的视觉语言模型。它擅长文档解析和结构化信息提取。

## 模型

- **模型 ID**：`mlx-community/GLM-OCR-bf16`
- **架构**：具有 M-RoPE（多维旋转位置嵌入）的视觉语言模型
- **支持的任务**：文本识别、公式识别、表格识别和结构化信息提取

## 安装

**使用 pip 安装：**
```sh
pip install mlx-vlm
```

**或使用 uv（快速 Python 包管理器）：**
```sh
uv pip install mlx-vlm
```

## 使用方法

### CLI

**基本文本识别：**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image document.png --prompt "Text Recognition:"
```

**公式识别：**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image equation.png --prompt "Formula Recognition:"
```

**表格识别：**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image table.png --prompt "Table Recognition:"
```

**结构化信息提取：**
```bash
uv run mlx_vlm generate --model mlx-community/GLM-OCR-bf16 --image id_card.png --prompt '请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}'
```

### Python 脚本

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# 加载模型
model, processor = load("mlx-community/GLM-OCR-bf16")

# 文档解析 - 文本识别
prompt = "Text Recognition:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["document.png"],
    max_tokens=512,
    verbose=True,
)
print(result.text)
```

**公式识别：**
```python
prompt = "Formula Recognition:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["equation.png"],
    max_tokens=256,
)
print(result.text)
```

**表格识别：**
```python
prompt = "Table Recognition:"
formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["table.png"],
    max_tokens=1024,
)
print(result.text)
```

**结构化信息提取：**
```python
# 定义用于提取的 JSON 模式
prompt = """请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}"""

formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)

result = generate(
    model,
    processor,
    formatted_prompt,
    image=["id_card.png"],
    max_tokens=512,
)
print(result.text)
```

## 支持的提示词

GLM-OCR 支持两种提示词场景：

### 1. 文档解析

使用这些任务提示词从文档中提取原始内容：

| 任务 | 提示词 |
|------|--------|
| 文本识别 | `Text Recognition:` |
| 公式识别 | `Formula Recognition:` |
| 表格识别 | `Table Recognition:` |

### 2. 信息提取

从文档中提取结构化信息。提示词必须遵循严格的 JSON 模式格式。

**示例 - 身份证提取：**
```
请按下列JSON格式输出图中信息:
{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}
```

**示例 - 发票提取：**
```
请按下列JSON格式输出图中信息:
{
    "invoice_number": "",
    "date": "",
    "vendor": "",
    "items": [],
    "subtotal": "",
    "tax": "",
    "total": ""
}
```

> **注意**：使用信息提取时，输出必须严格遵循定义的 JSON 模式，以确保下游处理的兼容性。

## 参数

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `max_tokens` | 要生成的最大 token 数 | 256 |
| `temperature` | 采样温度（0 = 确定性）| 0.0 |
| `top_p` | 核采样参数 | 1.0 |

## 示例：完整的 OCR 流程

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import json

# 加载一次模型
model, processor = load("mlx-community/GLM-OCR-bf16")

def extract_text(image_path: str) -> str:
    """从图像中提取原始文本。"""
    prompt = "Text Recognition:"
    formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)
    result = generate(model, processor, formatted_prompt, image=[image_path], max_tokens=1024)
    return result.text

def extract_structured(image_path: str, schema: dict) -> dict:
    """使用 JSON 模式提取结构化信息。"""
    prompt = f"请按下列JSON格式输出图中信息:\n{json.dumps(schema, indent=4, ensure_ascii=False)}"
    formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)
    result = generate(model, processor, formatted_prompt, image=[image_path], max_tokens=512)
    return json.loads(result.text)

# 使用
text = extract_text("document.png")
print(f"Extracted text: {text}")

# 结构化提取
schema = {
    "title": "",
    "author": "",
    "date": "",
    "content": ""
}
data = extract_structured("article.png", schema)
print(f"Extracted data: {data}")
```

## 致谢

此模型是 **[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR))** 的移植版本，由 [ZAI 团队](https://huggingface.co/zai-org) 开发。我们感谢 ZAI 团队在这个强大 OCR 模型上的工作。
