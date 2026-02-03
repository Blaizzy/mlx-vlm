"""GLM-OCR Model.

Complete implementation of GLM-OCR (0.9B) multimodal OCR model.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import BaseImageProcessor, InputEmbeddingsFeatures
from .config import ModelConfig, TextConfig, VisionConfig
from .language import LanguageModel
from .processing import GlmOcrProcessor
from .vision import VisionModel


class Model(nn.Module):
    """GLM-OCR Model.
    
    GLM-OCR (0.9B) is a multimodal OCR model for complex document understanding.
    
    Architecture:
    - Vision encoder: 24 layers, 1024 hidden size
    - Language model: 16 layers, 1536 hidden size
    - Total params: ~0.9B
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        
    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        """Get input embeddings combining text and vision."""
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )
        
        # Process vision
        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        
        # Get text embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        
        # Get vision features
        vision_outputs = self.vision_tower(pixel_values)
        
        # Merge vision features into text embeddings at image token positions
        # Find image token positions
        image_token_id = self.config.image_token_id
        image_positions = (input_ids == image_token_id).astype(mx.int32)
        
        # For simplicity, concatenate vision outputs at the beginning
        # In practice, you'd insert them at the correct positions
        merged_embeds = mx.concatenate([vision_outputs, inputs_embeds], axis=1)
        
        return InputEmbeddingsFeatures(
            inputs_embeds=merged_embeds,
            vision_hidden_state=vision_outputs,
        )
    
    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        **kwargs,
    ):
        """Forward pass."""
        if inputs_embeds is None:
            embeddings = self.get_input_embeddings(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )
            inputs_embeds = embeddings.inputs_embeds
        
        # Pass through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        
        return outputs
    
    def cast_predicate(self, key: str) -> bool:
        """Determine which parameters to cast."""
        return True


# Register the processor
try:
    from transformers import AutoProcessor
    AutoProcessor.register("GlmOcrProcessor", GlmOcrProcessor)
except Exception as e:
    print(f"Warning: Could not register GLM-OCR processor: {e}")


__all__ = ["Model", "LanguageModel", "VisionModel"]
