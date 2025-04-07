"""
Batch processor utilities to ensure all models work with batch_generate.
This module provides functions to fix processor configurations for various VLM models.
"""

import logging
import inspect
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def fix_processor_for_batch(processor, model_config):
    """
    Fix processor configuration for batch processing across all model types.
    
    Args:
        processor: The processor instance to fix
        model_config: The model configuration
        
    Returns:
        The fixed processor
    """
    # Fix for LlavaProcessor
    if hasattr(processor, 'tokenizer') and (not hasattr(processor, 'patch_size') or processor.patch_size is None):
        # Try to extract patch_size from model config
        patch_size = extract_patch_size(model_config)
        processor.patch_size = patch_size
        processor.vision_feature_select_strategy = "default"
        logger.info(f"Set LlavaProcessor patch_size={patch_size} and vision_feature_select_strategy='default'")
    
    # Fix specific issues for different model types
    model_type = get_model_type(model_config)
    
    if model_type == "llava":
        # Additional LLaVA-specific fixes if needed
        pass
    elif model_type == "qwen2-vl":
        # Qwen2-VL specific fixes if needed
        pass
    elif model_type == "pixtral":
        # Pixtral specific fixes if needed
        pass
    elif model_type == "deepseek-vl":
        # DeepSeek-VL specific fixes if needed
        pass
    
    return processor

def extract_patch_size(model_config) -> int:
    """
    Extract patch_size from model configuration.
    
    Args:
        model_config: The model configuration
        
    Returns:
        int: The patch size (defaults to 14 if not found)
    """
    # Try various configuration paths to find patch_size
    if hasattr(model_config, 'vision_config'):
        vision_config = model_config.vision_config
        # Try different attribute names for patch_size
        for attr in ['image_patch_size', 'patch_size', 'vision_patch_size']:
            if hasattr(vision_config, attr):
                return getattr(vision_config, attr)
                
    # Check if vision_config is a dictionary
    if isinstance(getattr(model_config, 'vision_config', None), dict):
        vision_dict = model_config.vision_config
        for key in ['image_patch_size', 'patch_size', 'vision_patch_size']:
            if key in vision_dict:
                return vision_dict[key]
        
        # Check nested params dict
        if 'params' in vision_dict:
            params = vision_dict['params']
            for key in ['image_patch_size', 'patch_size', 'vision_patch_size']:
                if key in params:
                    return params[key]
    
    # Default fallback value
    return 14

def get_model_type(model_config) -> str:
    """
    Determine the model type from configuration.
    
    Args:
        model_config: The model configuration
        
    Returns:
        str: The model type as a string
    """
    config_str = str(model_config).lower()
    
    if "llava" in config_str:
        return "llava"
    elif "qwen" in config_str:
        return "qwen2-vl"
    elif "pixtral" in config_str:
        return "pixtral"
    elif "deepseek" in config_str:
        return "deepseek-vl"
    else:
        return "unknown"

def batch_preprocess_images(processor, model_type, images):
    """
    Preprocess images for batch processing based on model type.
    
    Args:
        processor: The processor instance
        model_type: The model type string
        images: List of images to process
        
    Returns:
        List of processed images
    """
    # Some models require consistent image dimensions in batches
    if model_type == "pixtral":
        # Process pixtral images to have consistent dimensions
        from mlx_vlm.utils import load_image, process_image
        if images and all(isinstance(img, str) for img in images):
            return [process_image(load_image(img), (560, 560), None) for img in images]
    
    return images 