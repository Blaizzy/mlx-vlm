"""
Batch processing utilities for variable-sized images in MLX-VLM.

This module implements the transformers-style approach to handling batched 
generation with images of different sizes:

1. Group images by shape for efficient batch processing
2. Process each group as a batch (faster on GPU)
3. Pad on the patch/token dimension (not spatial dimensions)
4. Track original image sizes for attention masking
5. Reorder results back to original batch order

Key insight: Instead of resizing all images to the same dimensions (which
distorts aspect ratios), we let each image keep its natural size and pad
the sequence of patches to create uniform tensors for batching.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image


@dataclass
class ImageBatchInfo:
    """
    Information about a batch of processed images.

    Attributes:
        pixel_values: Batched pixel values tensor [batch_size, ...]
        image_sizes: Original (height, width) for each image
        num_patches: Number of actual patches/tokens per image
        attention_mask: Mask indicating real vs padded patches [batch_size, max_patches]
        original_indices: Mapping back to original batch order
    """

    pixel_values: mx.array
    image_sizes: List[Tuple[int, int]]
    num_patches: List[int]
    attention_mask: Optional[mx.array] = None
    original_indices: Optional[List[int]] = None


def group_images_by_shape(
    images: List[Image.Image],
    disable_grouping: bool = False,
) -> Tuple[Dict[Tuple[int, int], List[Image.Image]], Dict[Tuple[int, int], List[int]]]:
    """
    Group images by their dimensions for efficient batch processing.

    Images with the same dimensions can be stacked and processed together,
    which is much faster than processing individually (especially on GPU).

    Args:
        images: List of PIL images to group
        disable_grouping: If True, each image gets its own group (useful for debugging)

    Returns:
        grouped_images: Dict mapping shape -> list of images with that shape
        grouped_indices: Dict mapping shape -> list of original indices

    Example:
        >>> images = [img_400x300, img_800x600, img_400x300_2]
        >>> grouped, indices = group_images_by_shape(images)
        >>> grouped
        {(300, 400): [img_400x300, img_400x300_2], (600, 800): [img_800x600]}
        >>> indices
        {(300, 400): [0, 2], (600, 800): [1]}
    """
    if disable_grouping:
        # Each image in its own group
        grouped_images = {}
        grouped_indices = {}
        for i, img in enumerate(images):
            shape = (img.height, img.width)
            # Make each shape unique by adding index
            unique_shape = (img.height, img.width, i)
            grouped_images[unique_shape] = [img]
            grouped_indices[unique_shape] = [i]
        return grouped_images, grouped_indices

    grouped_images: Dict[Tuple[int, int], List[Image.Image]] = {}
    grouped_indices: Dict[Tuple[int, int], List[int]] = {}

    for i, img in enumerate(images):
        shape = (img.height, img.width)
        if shape not in grouped_images:
            grouped_images[shape] = []
            grouped_indices[shape] = []
        grouped_images[shape].append(img)
        grouped_indices[shape].append(i)

    return grouped_images, grouped_indices


def reorder_images(
    processed_groups: Dict[Tuple[int, int], mx.array],
    grouped_indices: Dict[Tuple[int, int], List[int]],
) -> List[mx.array]:
    """
    Reorder processed image groups back to original batch order.

    After processing groups of same-sized images, this function
    reorders them back to match the original input order.

    Args:
        processed_groups: Dict mapping shape -> processed tensors for that group
        grouped_indices: Dict mapping shape -> original indices (from group_images_by_shape)

    Returns:
        List of processed tensors in original batch order
    """
    # Determine total number of images
    total_images = sum(len(indices) for indices in grouped_indices.values())

    # Create output list
    result = [None] * total_images

    for shape, indices in grouped_indices.items():
        processed = processed_groups[shape]
        for i, original_idx in enumerate(indices):
            result[original_idx] = processed[i]

    return result


def compute_num_patches(
    image_size: Tuple[int, int],
    patch_size: int = 14,
    use_cls_token: bool = True,
) -> int:
    """
    Compute the number of patches/tokens for an image.

    Args:
        image_size: (height, width) of the image
        patch_size: Size of each patch (typically 14 or 16)
        use_cls_token: Whether to include CLS token in count

    Returns:
        Number of patches/tokens the image will produce
    """
    height, width = image_size
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    num_patches = num_patches_h * num_patches_w

    if use_cls_token:
        num_patches += 1

    return num_patches


def pad_patches_for_batching(
    patch_sequences: List[mx.array],
    pad_value: float = 0.0,
) -> Tuple[mx.array, mx.array, List[int]]:
    """
    Pad patch sequences to the same length for batching.

    This is the key technique: instead of padding spatial dimensions,
    we pad the sequence of patches. Each image keeps its natural
    aspect ratio and produces a different number of patches. We then
    pad the shorter sequences to match the longest.

    Args:
        patch_sequences: List of patch tensors, each [num_patches, hidden_dim]
        pad_value: Value to use for padding

    Returns:
        batched: Stacked tensor [batch_size, max_patches, hidden_dim]
        attention_mask: Binary mask [batch_size, max_patches] (1=real, 0=padded)
        num_patches: List of actual patch counts per image

    Example:
        >>> seq1 = mx.zeros((150, 768))  # Small image: 150 patches
        >>> seq2 = mx.zeros((600, 768))  # Large image: 600 patches
        >>> batched, mask, counts = pad_patches_for_batching([seq1, seq2])
        >>> batched.shape
        (2, 600, 768)
        >>> counts
        [150, 600]
        >>> mask[0, :150].sum(), mask[0, 150:].sum()  # First 150 real, rest padded
        (150, 0)
    """
    if not patch_sequences:
        raise ValueError("patch_sequences cannot be empty")

    # Get dimensions
    hidden_dim = patch_sequences[0].shape[-1]
    num_patches_list = [seq.shape[0] for seq in patch_sequences]
    max_patches = max(num_patches_list)
    batch_size = len(patch_sequences)

    # Create padded tensor
    batched = mx.full((batch_size, max_patches, hidden_dim), pad_value)
    attention_mask = mx.zeros((batch_size, max_patches))

    # Fill in actual values
    padded_list = []
    mask_list = []

    for i, (seq, num_patches) in enumerate(zip(patch_sequences, num_patches_list)):
        # Pad sequence
        if num_patches < max_patches:
            padding = mx.full((max_patches - num_patches, hidden_dim), pad_value)
            padded_seq = mx.concatenate([seq, padding], axis=0)
        else:
            padded_seq = seq
        padded_list.append(padded_seq)

        # Create mask (1 for real, 0 for padded)
        mask = mx.concatenate(
            [mx.ones((num_patches,)), mx.zeros((max_patches - num_patches,))]
        )
        mask_list.append(mask)

    batched = mx.stack(padded_list, axis=0)
    attention_mask = mx.stack(mask_list, axis=0)

    return batched, attention_mask, num_patches_list


def pad_pixel_values_for_batching(
    pixel_values_list: List[mx.array],
    pad_value: float = 0.0,
) -> Tuple[mx.array, List[Tuple[int, int]]]:
    """
    Pad pixel values tensors to create a uniform batch.

    For models that need spatial padding (when images have different
    spatial dimensions), this pads to the maximum height and width.

    Args:
        pixel_values_list: List of pixel tensors, each [C, H, W] or [num_patches, C, H, W]
        pad_value: Value to use for padding

    Returns:
        batched: Stacked tensor with uniform dimensions
        image_sizes: Original (height, width) for each image

    Note:
        This is the fallback approach. The preferred approach is to
        use pad_patches_for_batching on the patch embeddings instead.
    """
    if not pixel_values_list:
        raise ValueError("pixel_values_list cannot be empty")

    # Determine if we're dealing with patches or raw images
    first_shape = pixel_values_list[0].shape

    if len(first_shape) == 3:
        # Shape: [C, H, W] - raw images
        c = first_shape[0]
        max_h = max(pv.shape[1] for pv in pixel_values_list)
        max_w = max(pv.shape[2] for pv in pixel_values_list)

        image_sizes = [(pv.shape[1], pv.shape[2]) for pv in pixel_values_list]

        padded_list = []
        for pv in pixel_values_list:
            h, w = pv.shape[1], pv.shape[2]
            if h < max_h or w < max_w:
                # Pad to max dimensions
                padded = mx.full((c, max_h, max_w), pad_value)
                padded = padded.at[:, :h, :w].set(pv)
            else:
                padded = pv
            padded_list.append(padded)

        batched = mx.stack(padded_list, axis=0)

    elif len(first_shape) == 4:
        # Shape: [num_patches, C, H, W] - patch-based
        c, patch_h, patch_w = first_shape[1], first_shape[2], first_shape[3]
        max_patches = max(pv.shape[0] for pv in pixel_values_list)

        image_sizes = [
            (pv.shape[0],) for pv in pixel_values_list
        ]  # num_patches as "size"

        padded_list = []
        for pv in pixel_values_list:
            num_patches = pv.shape[0]
            if num_patches < max_patches:
                padding = mx.full(
                    (max_patches - num_patches, c, patch_h, patch_w), pad_value
                )
                padded = mx.concatenate([pv, padding], axis=0)
            else:
                padded = pv
            padded_list.append(padded)

        batched = mx.stack(padded_list, axis=0)
    else:
        raise ValueError(f"Unexpected pixel values shape: {first_shape}")

    return batched, image_sizes


def create_image_attention_mask(
    num_patches_list: List[int],
    max_patches: Optional[int] = None,
) -> mx.array:
    """
    Create attention mask for batched images with different numbers of patches.

    Args:
        num_patches_list: Number of actual patches per image
        max_patches: Maximum patches (inferred if not provided)

    Returns:
        Attention mask [batch_size, max_patches] where 1=attend, 0=ignore
    """
    if max_patches is None:
        max_patches = max(num_patches_list)

    batch_size = len(num_patches_list)
    mask = mx.zeros((batch_size, max_patches))

    mask_list = []
    for num_patches in num_patches_list:
        row_mask = mx.concatenate(
            [mx.ones((num_patches,)), mx.zeros((max_patches - num_patches,))]
        )
        mask_list.append(row_mask)

    return mx.stack(mask_list, axis=0)


def sort_images_by_size(
    images: List[Image.Image],
    descending: bool = True,
) -> Tuple[List[Image.Image], List[int]]:
    """
    Sort images by area (height * width) to minimize padding when batching.

    Processing similar-sized images together reduces padding waste.

    Args:
        images: List of PIL images
        descending: If True, largest images first

    Returns:
        sorted_images: Images sorted by size
        original_indices: Mapping to restore original order
    """
    areas = [img.height * img.width for img in images]
    sorted_indices = sorted(
        range(len(images)), key=lambda i: areas[i], reverse=descending
    )
    sorted_images = [images[i] for i in sorted_indices]

    return sorted_images, sorted_indices


def unsort_results(
    results: List,
    sorted_indices: List[int],
) -> List:
    """
    Restore results to original order after processing sorted images.

    Args:
        results: List of results in sorted order
        sorted_indices: Indices from sort_images_by_size

    Returns:
        Results in original order
    """
    unsorted = [None] * len(results)
    for sorted_pos, original_idx in enumerate(sorted_indices):
        unsorted[original_idx] = results[sorted_pos]
    return unsorted


class BatchImageProcessor:
    """
    High-level interface for batch processing variable-sized images.

    This class implements the full pipeline:
    1. Sort images by size (optional, for efficiency)
    2. Group by shape
    3. Process each group
    4. Pad for batching
    5. Track original sizes and create masks
    6. Reorder back to original order

    Example:
        >>> processor = BatchImageProcessor(patch_size=14)
        >>> images = [img1, img2, img3]  # Different sizes
        >>> batch_info = processor.process(images, preprocess_fn)
        >>> # batch_info.pixel_values: [3, max_patches, C, H, W]
        >>> # batch_info.image_sizes: [(h1, w1), (h2, w2), (h3, w3)]
        >>> # batch_info.attention_mask: [3, max_patches]
    """

    def __init__(
        self,
        patch_size: int = 14,
        use_cls_token: bool = True,
        disable_grouping: bool = False,
        sort_by_size: bool = True,
    ):
        """
        Initialize the batch processor.

        Args:
            patch_size: Size of each image patch
            use_cls_token: Whether the model uses CLS tokens
            disable_grouping: If True, process each image separately
            sort_by_size: If True, sort images by size before grouping
        """
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token
        self.disable_grouping = disable_grouping
        self.sort_by_size = sort_by_size

    def process(
        self,
        images: List[Image.Image],
        preprocess_fn,
    ) -> ImageBatchInfo:
        """
        Process a batch of variable-sized images.

        Args:
            images: List of PIL images (can have different sizes)
            preprocess_fn: Function to preprocess images -> pixel values

        Returns:
            ImageBatchInfo with batched tensors and metadata
        """
        if not images:
            raise ValueError("images list cannot be empty")

        batch_size = len(images)

        # Optionally sort by size
        if self.sort_by_size and batch_size > 1:
            sorted_images, sorted_indices = sort_images_by_size(images)
        else:
            sorted_images = images
            sorted_indices = list(range(batch_size))

        # Track original sizes
        image_sizes = [(img.height, img.width) for img in sorted_images]

        # Group by shape
        grouped_images, grouped_indices = group_images_by_shape(
            sorted_images, disable_grouping=self.disable_grouping
        )

        # Process each group
        processed_groups = {}
        for shape, group in grouped_images.items():
            # For same-sized images, batch process
            pixel_values = preprocess_fn(group)
            processed_groups[shape] = pixel_values

        # Reorder back
        processed_list = reorder_images(processed_groups, grouped_indices)

        # Pad for final batching
        batched_pixels, recorded_sizes = pad_pixel_values_for_batching(processed_list)

        # Compute patch counts and create attention mask
        num_patches = [
            compute_num_patches(size, self.patch_size, self.use_cls_token)
            for size in image_sizes
        ]
        attention_mask = create_image_attention_mask(num_patches)

        # Map back to original order
        final_indices = [sorted_indices[i] for i in range(batch_size)]

        return ImageBatchInfo(
            pixel_values=batched_pixels,
            image_sizes=image_sizes,
            num_patches=num_patches,
            attention_mask=attention_mask,
            original_indices=final_indices,
        )


def prepare_batched_inputs(
    images: List[Image.Image],
    processor,
    prompts: List[str],
    image_token_index: int,
    padding_side: str = "left",
) -> Dict:
    """
    Prepare batched inputs for VLM generation with variable-sized images.

    This is a convenience function that handles the full pipeline:
    1. Process images with grouping and padding
    2. Tokenize prompts
    3. Create attention masks
    4. Return ready-to-use inputs

    Args:
        images: List of PIL images (can have different sizes)
        processor: HuggingFace processor with tokenizer and image_processor
        prompts: List of text prompts
        image_token_index: Token ID for image placeholder
        padding_side: "left" or "right" for text padding

    Returns:
        Dict with input_ids, pixel_values, attention_mask, image_sizes, etc.
    """
    from .utils import process_inputs_with_fallback

    batch_size = len(images)

    if len(prompts) != batch_size:
        raise ValueError(
            f"Number of prompts ({len(prompts)}) must match number of images ({batch_size})"
        )

    # Sort images by area to group similar sizes
    sorted_indices = sorted(
        range(batch_size), key=lambda i: images[i].height * images[i].width
    )

    # Reorder images and prompts
    sorted_images = [images[i] for i in sorted_indices]
    sorted_prompts = [prompts[i] for i in sorted_indices]

    # Track original sizes before any processing
    image_sizes = [(img.height, img.width) for img in sorted_images]

    # Process inputs using existing infrastructure
    # This handles the tokenization and image preprocessing
    inputs = process_inputs_with_fallback(
        processor,
        prompts=sorted_prompts,
        images=sorted_images,
        audio=None,
        add_special_tokens=True,
        return_tensors="mlx",
    )

    # Add image sizes and sorted indices for later use
    inputs["image_sizes"] = image_sizes
    inputs["sorted_indices"] = sorted_indices

    return inputs


def restore_batch_order(
    results: List,
    sorted_indices: List[int],
) -> List:
    """
    Restore results to original batch order after processing.

    Args:
        results: Results in sorted order
        sorted_indices: Original indices from sorting

    Returns:
        Results in original batch order
    """
    restored = [None] * len(results)
    for sorted_pos, original_idx in enumerate(sorted_indices):
        restored[original_idx] = results[sorted_pos]
    return restored
