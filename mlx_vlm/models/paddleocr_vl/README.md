# PaddleOCR-VL

MLX implementation of [mlx-community/PaddleOCR-VL-1.5-bf16](https://huggingface.co/mlx-community/PaddleOCR-VL-1.5-bf16).

## Supported Tasks

- **OCR:** - Basic text recognition
- **Table Recognition:** - Table structure and content parsing
- **Formula Recognition:** - Mathematical formula detection
- **Chart Recognition:** - Chart parsing
- **Spotting:** - Text localization with bounding boxes
- **Seal Recognition:** - Seal detection

## Example: Text Spotting with Bounding Boxes

```python
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageDraw
import re

# Load model
model_path = 'mlx-community/PaddleOCR-VL-1.5-bf16'
model, processor = load(model_path)
config = load_config(model_path)

# Load image
image_path = 'your_image.jpg'
image = Image.open(image_path)
img_width, img_height = image.size

# Use Spotting prompt for bbox output
prompt = 'Spotting:'
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

# Generate
output = generate(model, processor, formatted_prompt, [image_path], max_tokens=8000, verbose=False)
text = output.text

# Parse bounding boxes from output
# Format: text<|LOC_x1|><|LOC_y1|><|LOC_x2|><|LOC_y2|><|LOC_x3|><|LOC_y3|><|LOC_x4|><|LOC_y4|>
pattern = r'([^<\n]+?)(<\|LOC_\d+\|><\|LOC_\d+\|><\|LOC_\d+\|><\|LOC_\d+\|><\|LOC_\d+\|><\|LOC_\d+\|><\|LOC_\d+\|><\|LOC_\d+\|>)'
matches = re.findall(pattern, text)

# Draw bounding boxes on image
draw = ImageDraw.Draw(image)
colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF', '#FFA500', '#800080']

for idx, (txt, locs) in enumerate(matches):
    coords = [int(x) for x in re.findall(r'<\|LOC_(\d+)\|>', locs)]
    if len(coords) == 8:
        # Coordinates are in 1000x1000 normalized space
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        scale_x = img_width / 1000
        scale_y = img_height / 1000
        points = [
            (x1 * scale_x, y1 * scale_y),
            (x2 * scale_x, y2 * scale_y),
            (x3 * scale_x, y3 * scale_y),
            (x4 * scale_x, y4 * scale_y),
        ]
        color = colors[idx % len(colors)]
        draw.polygon(points, outline=color, width=3)
        print(f"'{txt.strip()}' -> bbox: {coords}")

# Save result
image.save('output_with_bboxes.png')
```

## Output Format

The `Spotting:` prompt returns text with bounding box coordinates in the format:
```
text<|LOC_x1|><|LOC_y1|><|LOC_x2|><|LOC_y2|><|LOC_x3|><|LOC_y3|><|LOC_x4|><|LOC_y4|>
```

Where:
- Coordinates are normalized to a 1000x1000 space
- Each bbox has 4 corner points (quadrilateral)
- Scale coordinates by `(img_width/1000, img_height/1000)` to get pixel positions

## Example Output

```
'August 31,' -> bbox: [747, 11, 801, 11, 801, 28, 747, 28]
'2023' -> bbox: [764, 31, 788, 31, 788, 46, 764, 46]
'Cash and cash equivalents' -> bbox: [32, 102, 162, 102, 162, 119, 32, 119]
'11,613' -> bbox: [797, 103, 830, 103, 830, 120, 797, 120]
```

## CLI Usage

```bash
# Basic OCR
mlx_vlm.generate --model mlx-community/PaddleOCR-VL-1.5-bf16 --prompt "OCR:" --image image.jpg

# Table Recognition
mlx_vlm.generate --model mlx-community/PaddleOCR-VL-1.5-bf16 --prompt "Table Recognition:" --image table.jpg

# Text Spotting with bboxes
mlx_vlm.generate --model mlx-community/PaddleOCR-VL-1.5-bf16 --prompt "Spotting:" --image document.jpg --max-tokens 8000
```
