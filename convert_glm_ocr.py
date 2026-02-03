#!/usr/bin/env python3
"""
Script to convert GLM-OCR model from Hugging Face to MLX format.

Usage:
    python convert_glm_ocr.py --hf-path THUDM/glm-ocr-0.9b --mlx-path ./mlx-glm-ocr

Options:
    --quantize: Enable 4-bit quantization for smaller model size
    --q-group-size: Group size for quantization (default: 64)
    --q-bits: Bits per weight (default: 4)
    --upload-repo: HuggingFace repo to upload the converted model
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import mlx_vlm
sys.path.insert(0, str(Path(__file__).parent))

from mlx_vlm.convert import convert


def main():
    parser = argparse.ArgumentParser(
        description="Convert GLM-OCR model from HuggingFace to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default="./glm-ocr-hf",
        help="HuggingFace model path (default: ./glm-ocr-hf)",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx-glm-ocr",
        help="Output path for MLX model (default: mlx-glm-ocr)",
    )
    parser.add_argument(
        "-q", "--quantize",
        action="store_true",
        help="Quantize model to 4-bit (recommended for Mac with limited RAM)"
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)"
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Quantization bits (default: 4)"
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="HuggingFace repo to upload the converted model (optional)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code from HuggingFace (default: True)"
    )
    
    args = parser.parse_args()
    
    print(f"Converting GLM-OCR from {args.hf_path} to {args.mlx_path}")
    print(f"Quantization: {'enabled' if args.quantize else 'disabled'}")
    
    try:
        convert(
            hf_path=args.hf_path,
            mlx_path=args.mlx_path,
            quantize=args.quantize,
            q_group_size=args.q_group_size,
            q_bits=args.q_bits,
            upload_repo=args.upload_repo,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"\nConversion complete! Model saved to: {args.mlx_path}")
        print(f"\nTo use the model:")
        print(f"  python -m mlx_vlm.generate --model {args.mlx_path} --image <path> --prompt 'Extract text'")
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()
