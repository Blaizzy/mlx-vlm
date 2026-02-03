# GLM-OCR Model

GLM-OCR is a multimodal OCR model for complex document understanding, built on the GLM-V encoder-decoder architecture.

## Model Information

- **Model**: THUDM/glm-ocr-0.9b
- **Architecture**: GLM-V based with CogViT vision encoder
- **Parameters**: 0.9B
- **Specialization**: OCR, document understanding, layout analysis

## Features

- **Text Extraction**: Accurate text recognition from documents
- **Layout Understanding**: Tables, forms, and structured content
- **Formula Recognition**: Mathematical formulas in LaTeX notation
- **Code Recognition**: Code blocks and programming text
- **Multi-language Support**: Handles multiple languages in documents

## Usage

### Convert Model

First, convert the HuggingFace model to MLX format:

```bash
python convert_glm_ocr.py \
  --hf-path THUDM/glm-ocr-0.9b \
  --mlx-path ./mlx-glm-ocr \
  --quantize
```

Options:
- `--hf-path`: HuggingFace model path (default: THUDM/glm-ocr-0.9b)
- `--mlx-path`: Output path for MLX model
- `--quantize`: Enable 4-bit quantization (recommended for Mac with limited RAM)
- `--q-group-size`: Quantization group size (default: 64)
- `--q-bits`: Quantization bits (default: 4)

### CLI Inference

```bash
# Basic text extraction
mlx_vlm.generate \
  --model ./mlx-glm-ocr \
  --image ./document.png \
  --prompt "Extract all text from this image"

# Table extraction
mlx_vlm.generate \
  --model ./mlx-glm-ocr \
  --image ./table.png \
  --prompt "Extract this table in structured format"

# Formula recognition
mlx_vlm.generate \
  --model ./mlx-glm-ocr \
  --image ./math.png \
  --prompt "Extract and transcribe the mathematical formulas using LaTeX"
```

### Python API

```python
from mlx_vlm import load, generate

# Load model
model, processor = load("./mlx-glm-ocr")

# Generate text from image
output = generate(
    model,
    processor,
    image="./document.png",
    prompt="Extract all text from this image accurately",
    max_tokens=500,
    temperature=0.0,
)

print(output)
```

### OCR-Specific Prompts

The `GlmOcrProcessor` provides specialized prompts for different OCR tasks:

```python
from mlx_vlm.models.glm_ocr import GlmOcrProcessor

# Get OCR-specific prompt
processor = GlmOcrProcessor(...)
prompt = processor.get_ocr_prompt(task="table")

# Available tasks:
# - "extract_text": General text extraction
# - "table": Table extraction
# - "form": Form field extraction
# - "formula": Mathematical formula recognition
# - "document": Full document extraction
```

## Prompt Templates

### General Text Extraction
```
Extract all text content from this image accurately, preserving 
the layout and formatting as much as possible.
```

### Table Extraction
```
Extract the table from this image. Return the content in a structured 
format with rows and columns preserved.
```

### Form Extraction
```
Extract all fields and values from this form. Return as key-value pairs.
```

### Formula Recognition
```
Extract and transcribe any mathematical formulas or equations from 
this image using LaTeX notation.
```

## Performance

- **Model Size**: ~1.8GB (FP16), ~500MB (4-bit quantized)
- **Inference Speed**: Depends on image complexity and Mac hardware
- **Memory Requirements**: 8GB+ RAM recommended (4GB+ with quantization)

## Architecture Details

GLM-OCR inherits from GLM-4V architecture:

- **Vision Encoder**: CogViT with efficient token downsampling
- **Cross-Modal Connector**: Lightweight connector for vision-language fusion
- **Language Decoder**: GLM-0.5B text decoder
- **Special Tokens**: Uses GLM-4V tokenizer with vision tokens

## References

- [GLM-OCR HuggingFace](https://huggingface.co/THUDM/glm-ocr-0.9b)
- [GLM-OCR GitHub](https://github.com/zai-org/GLM-OCR)
- [GLM-4V Paper](https://arxiv.org/abs/2406.12749)

## License

Same as mlx-vlm package (MIT License)
