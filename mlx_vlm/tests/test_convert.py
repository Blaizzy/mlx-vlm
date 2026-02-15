"""Tests for convert.py functionality."""

import json


class TestSpatialMergeSizePreservation:
    """Test spatial_merge_size preservation during conversion.

    Background: Some processors (e.g., PixtralProcessor) don't serialize the
    spatial_merge_size field when calling save_pretrained(). This causes runtime
    errors like:
    "Number of image token positions (5476) does not match number of image features (1369)"

    The ratio 5476/1369 = 4 = spatial_merge_size^2, indicating the field is missing.

    See: https://github.com/huggingface/transformers/pull/37019

    Affected models include:
    - mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit
    - Other Pixtral-based models converted with mlx-vlm 0.1.19
    """

    def test_spatial_merge_size_copied_when_missing(self, tmp_path):
        """Test that spatial_merge_size is copied from config to processor_config."""
        # Setup: config.json has spatial_merge_size, processor_config.json doesn't
        config = {
            "model_type": "mistral3",
            "spatial_merge_size": 2,
            "vision_config": {"model_type": "pixtral"},
        }

        proc_config = {
            "image_token": "[IMG]",
            "image_break_token": "[IMG_BREAK]",
            "image_end_token": "[IMG_END]",
            "patch_size": 14,
            "processor_class": "PixtralProcessor",
        }

        mlx_path = tmp_path / "mlx_model"
        mlx_path.mkdir()

        proc_config_path = mlx_path / "processor_config.json"
        with open(proc_config_path, "w") as f:
            json.dump(proc_config, f, indent=2)

        # Execute: Run the fix logic (extracted from convert.py)
        if "spatial_merge_size" in config:
            if proc_config_path.exists():
                with open(proc_config_path) as f:
                    loaded_proc_config = json.load(f)
                if (
                    loaded_proc_config.get("spatial_merge_size")
                    != config["spatial_merge_size"]
                ):
                    loaded_proc_config["spatial_merge_size"] = config[
                        "spatial_merge_size"
                    ]
                    with open(proc_config_path, "w") as f:
                        json.dump(loaded_proc_config, f, indent=2)

        # Verify: processor_config.json now has spatial_merge_size
        with open(proc_config_path) as f:
            result = json.load(f)

        assert "spatial_merge_size" in result
        assert result["spatial_merge_size"] == 2

    def test_spatial_merge_size_corrected_when_wrong_default(self, tmp_path):
        """Test that wrong spatial_merge_size default (1) is corrected to config value (2)."""
        config = {
            "model_type": "mistral3",
            "spatial_merge_size": 2,
        }

        # processor_config has wrong default value (1 instead of 2)
        proc_config = {
            "image_token": "[IMG]",
            "spatial_merge_size": 1,  # Wrong default from processor.save_pretrained()
        }

        mlx_path = tmp_path / "mlx_model"
        mlx_path.mkdir()

        proc_config_path = mlx_path / "processor_config.json"
        with open(proc_config_path, "w") as f:
            json.dump(proc_config, f, indent=2)

        # Execute: Run the fix logic (updated version)
        if "spatial_merge_size" in config:
            if proc_config_path.exists():
                with open(proc_config_path) as f:
                    loaded_proc_config = json.load(f)
                if (
                    loaded_proc_config.get("spatial_merge_size")
                    != config["spatial_merge_size"]
                ):
                    loaded_proc_config["spatial_merge_size"] = config[
                        "spatial_merge_size"
                    ]
                    with open(proc_config_path, "w") as f:
                        json.dump(loaded_proc_config, f, indent=2)

        # Verify: Value is corrected to match config.json
        with open(proc_config_path) as f:
            result = json.load(f)

        assert result["spatial_merge_size"] == 2  # Corrected from 1 to 2

    def test_no_action_when_config_lacks_spatial_merge_size(self, tmp_path):
        """Test that nothing happens if config doesn't have spatial_merge_size."""
        config = {
            "model_type": "llama",
            # No spatial_merge_size - not a vision model
        }

        proc_config = {
            "tokenizer_class": "LlamaTokenizer",
        }

        mlx_path = tmp_path / "mlx_model"
        mlx_path.mkdir()

        proc_config_path = mlx_path / "processor_config.json"
        with open(proc_config_path, "w") as f:
            json.dump(proc_config, f, indent=2)

        original_content = proc_config_path.read_text()

        # Execute: Run the fix logic
        if "spatial_merge_size" in config:
            if proc_config_path.exists():
                with open(proc_config_path) as f:
                    loaded_proc_config = json.load(f)
                if (
                    loaded_proc_config.get("spatial_merge_size")
                    != config["spatial_merge_size"]
                ):
                    loaded_proc_config["spatial_merge_size"] = config[
                        "spatial_merge_size"
                    ]
                    with open(proc_config_path, "w") as f:
                        json.dump(loaded_proc_config, f, indent=2)

        # Verify: File unchanged
        with open(proc_config_path) as f:
            result = json.load(f)

        assert "spatial_merge_size" not in result

    def test_no_action_when_processor_config_missing(self, tmp_path):
        """Test graceful handling when processor_config.json doesn't exist."""
        config = {
            "model_type": "mistral3",
            "spatial_merge_size": 2,
        }

        mlx_path = tmp_path / "mlx_model"
        mlx_path.mkdir()

        proc_config_path = mlx_path / "processor_config.json"
        # Don't create the file

        # Execute: Should not raise
        if "spatial_merge_size" in config:
            if proc_config_path.exists():
                with open(proc_config_path) as f:
                    loaded_proc_config = json.load(f)
                if (
                    loaded_proc_config.get("spatial_merge_size")
                    != config["spatial_merge_size"]
                ):
                    loaded_proc_config["spatial_merge_size"] = config[
                        "spatial_merge_size"
                    ]
                    with open(proc_config_path, "w") as f:
                        json.dump(loaded_proc_config, f, indent=2)

        # Verify: File still doesn't exist (no error thrown)
        assert not proc_config_path.exists()

    def test_no_action_when_values_match(self, tmp_path):
        """Test that no write happens if values already match."""
        config = {
            "model_type": "mistral3",
            "spatial_merge_size": 2,
        }

        proc_config = {
            "image_token": "[IMG]",
            "spatial_merge_size": 2,  # Already correct
        }

        mlx_path = tmp_path / "mlx_model"
        mlx_path.mkdir()

        proc_config_path = mlx_path / "processor_config.json"
        with open(proc_config_path, "w") as f:
            json.dump(proc_config, f, indent=2)

        import os

        original_mtime = os.path.getmtime(proc_config_path)

        # Small delay to ensure mtime would change if file is written
        import time

        time.sleep(0.01)

        # Execute: Run the fix logic
        if "spatial_merge_size" in config:
            if proc_config_path.exists():
                with open(proc_config_path) as f:
                    loaded_proc_config = json.load(f)
                if (
                    loaded_proc_config.get("spatial_merge_size")
                    != config["spatial_merge_size"]
                ):
                    loaded_proc_config["spatial_merge_size"] = config[
                        "spatial_merge_size"
                    ]
                    with open(proc_config_path, "w") as f:
                        json.dump(loaded_proc_config, f, indent=2)

        # Verify: File was not rewritten (mtime unchanged)
        assert os.path.getmtime(proc_config_path) == original_mtime
