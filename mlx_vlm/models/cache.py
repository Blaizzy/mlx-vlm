import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
from safetensors.torch import load_file, save_file


class VLMFeatureCache:
    """Cache for storing and retrieving image features from Vision Language Models."""

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize the feature cache.

        Args:
            cache_dir: Directory to store the cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """
        Compute SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: Hex digest of the file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """
        Get the cache file path for a given file hash.

        Args:
            file_hash: SHA-256 hash of the original file

        Returns:
            Path: Path where the cached features should be stored
        """
        return self.cache_dir / f"{file_hash}.safetensors"

    def save_features(
        self,
        image_path: Union[str, Path],
        features: Dict[str, mx.array],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save image features to cache.

        Args:
            image_path: Path to the original image file
            features: Dictionary of feature tensors to cache
            metadata: Optional metadata to store with the features

        Returns:
            str: Hash of the cached file
        """
        file_hash = self._compute_file_hash(image_path)
        cache_path = self._get_cache_path(file_hash)

        # Add original file path to metadata
        if metadata is None:
            metadata = {}
        metadata["original_file"] = str(image_path)
        metadata["format"] = "mlx"

        # Save features using safetensors
        mx.save_safetensors(str(cache_path), {"image_features": features}, metadata)
        return file_hash

    def load_features(
        self, image_path: Union[str, Path]
    ) -> Optional[Dict[str, mx.array]]:
        """
        Load cached features for an image if they exist.

        Args:
            image_path: Path to the image file

        Returns:
            Optional[Dict[str, mx.array]]: Cached features if they exist, None otherwise
        """
        file_hash = self._compute_file_hash(image_path)
        cache_path = self._get_cache_path(file_hash)

        if not cache_path.exists():
            return None

        features = mx.load(str(cache_path))
        return features

    def get_metadata(self, image_path: Union[str, Path]) -> Optional[Dict[str, str]]:
        """
        Get metadata for cached features if they exist.

        Args:
            image_path: Path to the image file

        Returns:
            Optional[Dict[str, str]]: Metadata if cached features exist, None otherwise
        """
        file_hash = self._compute_file_hash(image_path)
        cache_path = self._get_cache_path(file_hash)

        if not cache_path.exists():
            return None

        return load_file(cache_path)

    def clear_cache(self):
        """Remove all cached features."""
        for cache_file in self.cache_dir.glob("*.safetensors"):
            cache_file.unlink()

    def get_cache_size(self) -> int:
        """
        Get the total size of cached features in bytes.

        Returns:
            int: Total size of cache in bytes
        """
        return (
            sum(f.stat().st_size for f in self.cache_dir.glob("*.safetensors"))
            / (1024 * 1024 * 1024),
            "GB",
        )

    def __contains__(self, image_path: Union[str, Path]) -> bool:
        """
        Check if features for an image are cached.

        Args:
            image_path: Path to the image file

        Returns:
            bool: True if features are cached, False otherwise
        """
        file_hash = self._compute_file_hash(image_path)
        return self._get_cache_path(file_hash).exists()
