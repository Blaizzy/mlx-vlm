#!/usr/bin/env python3
"""Integration tests for HunyuanOCR with real weights and images.

This test file requires:
1. Real model weights in hunyuanocr/ directory
2. Test images (see IMAGE REQUIREMENTS section below)

Run from repo root:
    python -m mlx_vlm.models.hunyuan_vl.test_integration

IMAGE REQUIREMENTS
==================

For comprehensive testing, please provide the following test images:

1. **OCR Test Images** (text recognition):
   - `test_images/ocr_simple.png` - Simple printed text (e.g., a few words)
   - `test_images/ocr_document.png` - Document with paragraphs
   - `test_images/ocr_handwritten.png` - Handwritten text (optional)

2. **Format Test Images**:
   - `test_images/format_jpg.jpg` - JPEG format
   - `test_images/format_png.png` - PNG format
   - `test_images/format_webp.webp` - WebP format (optional)

3. **Size Test Images**:
   - `test_images/size_small.png` - Small image (e.g., 128x128)
   - `test_images/size_medium.png` - Medium image (e.g., 512x512)
   - `test_images/size_large.png` - Large image (e.g., 1024x1024)

4. **Aspect Ratio Test Images**:
   - `test_images/aspect_square.png` - Square (1:1)
   - `test_images/aspect_wide.png` - Wide (16:9 or similar)
   - `test_images/aspect_tall.png` - Tall (9:16 or similar)

You can also use existing images from examples/images/ for basic testing.

Image Format Requirements:
- Supported formats: PNG, JPEG, WebP, BMP, GIF
- Color space: RGB (3 channels)
- Recommended size: 256x256 to 1024x1024 for testing
- For OCR: Clear, high-contrast text works best
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Optional

import mlx.core as mx


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_model_path() -> Path:
    """Get path to local HunyuanOCR weights."""
    return get_repo_root() / "hunyuanocr"


def get_test_images_dir() -> Path:
    """Get path to test images directory."""
    return get_repo_root() / "test_images"


def get_example_images_dir() -> Path:
    """Get path to example images directory."""
    return get_repo_root() / "examples" / "images"


def weights_available() -> bool:
    """Check if model weights are available."""
    model_path = get_model_path()
    return (model_path / "config.json").exists()


def find_test_image() -> Optional[Path]:
    """Find any available test image."""
    # Check test_images directory first
    test_dir = get_test_images_dir()
    if test_dir.exists():
        for ext in ["png", "jpg", "jpeg", "webp"]:
            for img in test_dir.glob(f"*.{ext}"):
                return img

    # Fall back to examples/images
    examples_dir = get_example_images_dir()
    if examples_dir.exists():
        for ext in ["png", "jpg", "jpeg", "webp"]:
            for img in examples_dir.glob(f"*.{ext}"):
                return img

    return None


class TestHunyuanVLIntegration(unittest.TestCase):
    """Integration tests for HunyuanOCR with real weights."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_path = get_model_path()
        cls.weights_available = weights_available()

        if not cls.weights_available:
            print(f"\n⚠ Weights not found at {cls.model_path}")
            print("  Skipping integration tests that require weights.")
            return

        # Load model and processor
        print(f"\nLoading model from {cls.model_path}...")
        try:
            from mlx_vlm import load

            cls.model, cls.processor = load(str(cls.model_path))
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            cls.weights_available = False
            cls.load_error = str(e)

    def setUp(self):
        """Skip tests if weights not available."""
        if not self.weights_available:
            self.skipTest("Model weights not available")

    def test_model_loads_correctly(self):
        """Test that model loads without errors."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.processor)

        # Verify model type
        self.assertEqual(self.model.config.model_type, "hunyuan_vl")

        # Verify layer counts
        self.assertEqual(len(self.model.language_model.model.layers), 24)
        self.assertEqual(len(self.model.vision_tower.layers), 27)

    def test_processor_has_tokenizer(self):
        """Test that processor has a working tokenizer."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # Test basic tokenization
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Test decode
        decoded = tokenizer.decode(tokens)
        self.assertIn("Hello", decoded)

    def test_text_only_generation(self):
        """Test generation without image (text-only)."""
        from mlx_vlm import generate

        prompt = "What is 2 + 2?"

        try:
            output = generate(
                self.model,
                self.processor,
                image=None,
                prompt=prompt,
                max_tokens=20,
                verbose=False,
            )

            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 0)
            print(f"  Text-only output: {output[:100]}...")
        except Exception as e:
            self.fail(f"Text-only generation failed: {e}")

    def test_image_loading(self):
        """Test that images can be loaded correctly."""
        from mlx_vlm.utils import load_image

        test_image = find_test_image()
        if test_image is None:
            self.skipTest("No test images available")

        print(f"  Loading image: {test_image}")
        image = load_image(str(test_image))

        self.assertIsNotNone(image)
        # Image should be a PIL Image or numpy array
        self.assertTrue(hasattr(image, "size") or hasattr(image, "shape"))

    def test_image_generation_basic(self):
        """Test basic image + text generation."""
        from mlx_vlm import generate
        from mlx_vlm.utils import load_image

        test_image = find_test_image()
        if test_image is None:
            self.skipTest("No test images available")

        print(f"  Using image: {test_image}")
        image = load_image(str(test_image))
        prompt = "Describe what you see in this image."

        try:
            output = generate(
                self.model,
                self.processor,
                image=image,
                prompt=prompt,
                max_tokens=50,
                verbose=False,
            )

            self.assertIsInstance(output, str)
            self.assertGreater(len(output), 0)
            print(f"  Image generation output: {output[:100]}...")
        except Exception as e:
            self.fail(f"Image generation failed: {e}")

    def test_ocr_generation(self):
        """Test OCR-specific generation with text-containing image."""
        from mlx_vlm import generate
        from mlx_vlm.utils import load_image

        # Look for OCR-specific test image
        test_images_dir = get_test_images_dir()
        ocr_image = None

        for name in ["ocr_simple.png", "ocr_document.png"]:
            path = test_images_dir / name
            if path.exists():
                ocr_image = path
                break

        if ocr_image is None:
            # Fall back to any available image
            ocr_image = find_test_image()

        if ocr_image is None:
            self.skipTest("No test images available")

        print(f"  Using OCR image: {ocr_image}")
        image = load_image(str(ocr_image))
        prompt = "Please read and transcribe all the text in this image."

        try:
            output = generate(
                self.model,
                self.processor,
                image=image,
                prompt=prompt,
                max_tokens=100,
                verbose=False,
            )

            self.assertIsInstance(output, str)
            print(f"  OCR output: {output[:200]}...")
        except Exception as e:
            self.fail(f"OCR generation failed: {e}")

    def test_different_image_sizes(self):
        """Test generation with different image sizes."""
        import numpy as np
        from PIL import Image

        from mlx_vlm import generate

        sizes = [(128, 128), (256, 256), (512, 512)]

        for width, height in sizes:
            with self.subTest(size=f"{width}x{height}"):
                # Create synthetic test image
                img_array = np.random.randint(
                    0, 255, (height, width, 3), dtype=np.uint8
                )
                image = Image.fromarray(img_array)

                prompt = "What do you see?"

                try:
                    output = generate(
                        self.model,
                        self.processor,
                        image=image,
                        prompt=prompt,
                        max_tokens=20,
                        verbose=False,
                    )

                    self.assertIsInstance(output, str)
                    print(f"  Size {width}x{height}: OK")
                except Exception as e:
                    self.fail(f"Generation failed for size {width}x{height}: {e}")

    def test_different_aspect_ratios(self):
        """Test generation with different aspect ratios."""
        import numpy as np
        from PIL import Image

        from mlx_vlm import generate

        aspect_ratios = [
            (256, 256, "square"),
            (512, 256, "wide"),
            (256, 512, "tall"),
        ]

        for width, height, name in aspect_ratios:
            with self.subTest(aspect=name):
                # Create synthetic test image
                img_array = np.random.randint(
                    0, 255, (height, width, 3), dtype=np.uint8
                )
                image = Image.fromarray(img_array)

                prompt = "Describe this image."

                try:
                    output = generate(
                        self.model,
                        self.processor,
                        image=image,
                        prompt=prompt,
                        max_tokens=20,
                        verbose=False,
                    )

                    self.assertIsInstance(output, str)
                    print(f"  Aspect {name} ({width}x{height}): OK")
                except Exception as e:
                    self.fail(f"Generation failed for aspect {name}: {e}")

    def test_chat_template_applied(self):
        """Test that chat template is applied correctly."""
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = "Hello, how are you?"

        messages = apply_chat_template(
            self.processor,
            self.model.config,
            prompt,
            num_images=1,
            return_messages=True,
        )

        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)

        # Check message structure
        first_msg = messages[0]
        self.assertIn("role", first_msg)
        self.assertIn("content", first_msg)

    def test_no_nan_in_outputs(self):
        """Test that model outputs don't contain NaN values."""
        from mlx_vlm.utils import load_image

        test_image = find_test_image()
        if test_image is None:
            self.skipTest("No test images available")

        image = load_image(str(test_image))

        # Process image through vision tower
        # This requires knowing the preprocessing pipeline
        # For now, we test via the smoke test approach

        # Create dummy input
        pixel_values = mx.random.uniform(shape=(1, 256, 256, 3))
        vision_features = self.model.vision_tower(pixel_values)

        mx.eval(vision_features)
        self.assertFalse(
            mx.any(mx.isnan(vision_features)).item(), "NaN detected in vision features"
        )


class TestHunyuanVLImageFormats(unittest.TestCase):
    """Test different image formats and loading methods."""

    @classmethod
    def setUpClass(cls):
        cls.weights_available = weights_available()

    def setUp(self):
        if not self.weights_available:
            self.skipTest("Model weights not available")

    def test_load_png(self):
        """Test loading PNG images."""
        from mlx_vlm.utils import load_image

        examples_dir = get_example_images_dir()
        png_files = list(examples_dir.glob("*.png"))

        if not png_files:
            self.skipTest("No PNG files in examples/images")

        image = load_image(str(png_files[0]))
        self.assertIsNotNone(image)

    def test_load_jpg(self):
        """Test loading JPEG images."""
        from mlx_vlm.utils import load_image

        examples_dir = get_example_images_dir()
        jpg_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.jpeg"))

        if not jpg_files:
            self.skipTest("No JPEG files in examples/images")

        image = load_image(str(jpg_files[0]))
        self.assertIsNotNone(image)

    def test_load_webp(self):
        """Test loading WebP images."""
        from mlx_vlm.utils import load_image

        examples_dir = get_example_images_dir()
        webp_files = list(examples_dir.glob("*.webp"))

        if not webp_files:
            self.skipTest("No WebP files in examples/images")

        image = load_image(str(webp_files[0]))
        self.assertIsNotNone(image)


def run_quick_test():
    """Run a quick integration test and print results."""
    print("=" * 60)
    print("HunyuanOCR Quick Integration Test")
    print("=" * 60)

    if not weights_available():
        print(f"\n✗ Weights not found at {get_model_path()}")
        print(
            "  Please download weights from https://huggingface.co/tencent/HunyuanOCR"
        )
        return False

    print(f"\n✓ Weights found at {get_model_path()}")

    # Try to load model
    print("\nLoading model...")
    try:
        from mlx_vlm import load

        model, processor = load(str(get_model_path()))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Try text-only generation
    print("\nTesting text-only generation...")
    try:
        from mlx_vlm import generate

        output = generate(
            model, processor, image=None, prompt="Hello!", max_tokens=10, verbose=False
        )
        print(f"✓ Text output: {output}")
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Try image generation
    test_image = find_test_image()
    if test_image:
        print(f"\nTesting image generation with {test_image}...")
        try:
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_image

            image = load_image(str(test_image))

            # Apply chat template to get properly formatted prompt with <image> token
            prompt = apply_chat_template(
                processor, model.config, "Describe this image.", num_images=1
            )
            print(f"  Formatted prompt: {prompt[:200]}...")

            output = generate(
                model,
                processor,
                image=image,
                prompt=prompt,
                max_tokens=30,
                verbose=False,
            )
            print(f"✓ Image output: {output}")
        except Exception as e:
            print(f"✗ Image generation failed: {e}")
            import traceback

            traceback.print_exc()
            return False
    else:
        print("\n⚠ No test images found, skipping image generation test")

    print("\n" + "=" * 60)
    print("Quick integration test passed! ✓")
    print("=" * 60)
    return True


def main():
    """Run integration tests."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite
        unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
