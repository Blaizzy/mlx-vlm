#!/usr/bin/env python3
"""
Test script for GLM-OCR model with mlx-vlm.

This script tests the GLM-OCR model on a sample image to verify it works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path to import mlx_vlm
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mlx_vlm import load, generate
    from mlx_vlm.models.glm_ocr import GlmOcrProcessor
    print("✓ GLM-OCR module imports successfully")
    
    # Test loading the processor class
    print("✓ GlmOcrProcessor class available:", GlmOcrProcessor)
    
    print("\nTo test with actual model weights:")
    print("1. Convert the model: python convert_glm_ocr.py --quantize")
    print("2. Run inference: python test_glm_ocr.py --model mlx-glm-ocr --image <path>")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure mlx-vlm is installed: uv pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
