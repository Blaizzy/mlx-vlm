# DOTS OCR

DOTS 模型是用于文档解析、布局分析和结构化提取的视觉语言 OCR 模型。`dots.mocr` 扩展了原始的 `dots.ocr` 检查点，具有更强的多语言解析和结构化图形生成能力。

## 模型

- **主要用例**：OCR、布局解析、表格提取、公式提取、结构化 JSON 输出

## 安装

```bash
uv pip install mlx-vlm
```

## CLI 示例

### 1) 布局 JSON 提取（详细提示）

```bash
uv run mlx_vlm.generate --model rednote-hilab/dots.mocr --prompt "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object." --image "path_to_image.jpg" --max-tokens 5000
```

### 2) 基本 OCR

```bash
uv run mlx_vlm.generate \
  --model rednote-hilab/dots.mocr \
  --image receipt.jpg \
  --prompt "Extract all text from this image." \
  --max-tokens 1024
```

### 3) Markdown 文档转换

```bash
uv run mlx_vlm.generate \
  --model mlx-community/dots.mocr-4bit \
  --image page.png \
  --prompt "Convert this page to clean Markdown while preserving reading order." \
  --max-tokens 4096
```

## Python 脚本示例

### 1) `layout_json.py`

```python

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.mocr-4bit"
IMAGE_PATH = "path_to_image.jpg"

PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""

model, processor = load(MODEL)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=IMAGE_PATH,
    max_tokens=5000,
    temperature=0.0,
)
print(result.text)
```

### 2) `basic_ocr.py`

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.mocr-4bit"
IMAGE_PATH = "receipt.jpg"
PROMPT = "Extract all text from this image."

model, processor = load(MODEL)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=IMAGE_PATH,
    max_tokens=1024,
    temperature=0.0,
)
print(result.text)
```

### 3) `markdown_document_conversion.py`
```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "mlx-community/dots.mocr-4bit"
IMAGE_PATH = "page.png"
PROMPT = "Convert this page to clean Markdown while preserving reading order."

model, processor = load(MODEL)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)

result = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=IMAGE_PATH,
    max_tokens=4096,
    temperature=0.0,
)
print(result.text)
```

## Notebook 演示

- [`examples/dots_mocr_demo.ipynb`](../../../examples/dots_mocr_demo.ipynb) 仅使用 MLX-VLM 运行独特的上游 `dots.mocr` README 场景，使用捆绑的本地演示资源，并保存原始输出、叠加层、渲染的 SVG 预览和联系表。无效的 SVG 生成会回退到可读的文本叠加层，而不是渲染器错误页面。
- 该 notebook 的捆绑源资源位于 [`examples/images`](../../../examples/images)。
- 该 notebook 为 `demo_hf_layout` 和 `parser_image_default` 创建别名文件，因为这些上游示例重用与主要文档解析运行相同的图像和提示。
- Notebook 中的 SVG 预览渲染使用 macOS `qlmanage`。

## 注意事项

- 对于长文档和布局密集的页面，增加 `--max-tokens`。
- 要强制严格的结构化输出，使提示对模式和排序具有明确性。
- 官方 `dots.ocr` README：https://huggingface.co/rednote-hilab/dots.ocr
- 官方 `dots.mocr` README：https://huggingface.co/rednote-hilab/dots.mocr
- DOTS 项目/博客：https://github.com/rednote-hilab/dots.mocr
