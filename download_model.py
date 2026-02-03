#!/usr/bin/env python3
"""Download GLM-OCR model weights from HuggingFace."""

from huggingface_hub import snapshot_download
import sys

print("Downloading GLM-OCR model weights from THUDM/glm-ocr-0.9b...")
print("This may take a few minutes (model is ~1-2GB)...")

try:
    model_path = snapshot_download(
        repo_id="zai-org/GLM-OCR",
        local_dir="./glm-ocr-hf",
        resume_download=True
    )
    print(f"\n✅ Model downloaded successfully to: {model_path}")
    print("\nNext step: Convert to MLX format")
    print(f"  python convert_glm_ocr.py --hf-path {model_path} --mlx-path ./mlx-glm-ocr --quantize")
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    sys.exit(1)
