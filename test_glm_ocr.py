#!/usr/bin/env python3
"""
Test script for GLM-OCR model with mlx-vlm.

This script tests the GLM-OCR model implementation without requiring
actual model weights (useful for CI/smoke testing).
"""

import sys
from pathlib import Path

# Add parent directory to path to import mlx_vlm
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all GLM-OCR modules can be imported."""
    print("Testing GLM-OCR module imports...")
    
    try:
        # Test importing the glm_ocr module
        from mlx_vlm.models import glm_ocr
        print("✓ mlx_vlm.models.glm_ocr imports successfully")
        
        # Test importing individual components
        from mlx_vlm.models.glm_ocr import Model, ModelConfig, GlmOcrProcessor
        print("✓ Model, ModelConfig, GlmOcrProcessor import successfully")
        
        # Test importing from glm4v (shared components)
        from mlx_vlm.models.glm_ocr import LanguageModel, VisionModel, TextConfig, VisionConfig
        print("✓ Shared components (LanguageModel, VisionModel, TextConfig, VisionConfig) import successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_model_config():
    """Test ModelConfig creation."""
    print("\nTesting ModelConfig...")
    
    try:
        from mlx_vlm.models.glm_ocr import ModelConfig, TextConfig, VisionConfig
        
        # Create minimal config
        text_config = TextConfig()
        vision_config = VisionConfig(
            model_type="glm4v_vision",
            depth=28,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            patch_size=14,
        )
        
        config = ModelConfig(
            text_config=text_config,
            vision_config=vision_config,
            model_type="glm_ocr",
        )
        
        assert config.model_type == "glm_ocr", "model_type should be 'glm_ocr'"
        assert config.vocab_size == 151552, "vocab_size should match GLM-4V"
        
        print(f"✓ ModelConfig created successfully")
        print(f"  - model_type: {config.model_type}")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - hidden_size: {config.hidden_size}")
        
        return True
    except Exception as e:
        print(f"✗ ModelConfig error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_processor():
    """Test GlmOcrProcessor creation and basic functionality."""
    print("\nTesting GlmOcrProcessor...")
    
    try:
        from mlx_vlm.models.glm_ocr import GlmOcrProcessor
        
        # Test that processor class exists and has expected methods
        assert hasattr(GlmOcrProcessor, 'get_ocr_prompt'), "Processor should have get_ocr_prompt method"
        
        print("✓ GlmOcrProcessor has expected methods")
        
        # Test OCR prompt generation
        # Note: We can't fully test without a tokenizer, but we can test the method exists
        print("✓ OCR prompt methods available")
        
        return True
    except Exception as e:
        print(f"✗ Processor error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils_integration():
    """Test integration with mlx_vlm utils."""
    print("\nTesting integration with mlx_vlm utils...")
    
    try:
        from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args
        
        # The model should be loadable via its model_type
        # Note: This will fail without actual weights, but tests the import path
        print("✓ Utils import successfully")
        
        # Check that glm_ocr model_type is not remapped
        if "glm_ocr" in MODEL_REMAPPING:
            print(f"  Note: glm_ocr is remapped to {MODEL_REMAPPING['glm_ocr']}")
        else:
            print("  Note: glm_ocr uses direct model_type mapping")
        
        return True
    except Exception as e:
        print(f"✗ Utils integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GLM-OCR Model Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_config,
        test_processor,
        test_utils_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ Test {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! GLM-OCR implementation looks good.")
        print("\nNext steps:")
        print("1. Convert model weights: python convert_glm_ocr.py --quantize")
        print("2. Test with real inference on a document image")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
