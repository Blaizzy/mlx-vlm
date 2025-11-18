from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_omni_moe.thinker import Thinker
from mlx_vlm.models.qwen3_omni_moe.talker import Talker
from mlx_vlm.models.qwen3_omni_moe.code2wav import Code2WavModel

from .config import ModelConfig

def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    # Reshape the tensors to 1D
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    # Scatter the scaled image features into the special image token positions
    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    # Reshape back to the original shape
    final_embedding = mx.reshape(final_embedding_flattened, final_embedding_shape)

    return final_embedding

"""
MOE模型的顶层总结构:
- thinker_model （除音频输入外，与 qwen3vlmoe 一致）
    - vision_tower
    - language_model
    - audio_tower
- talker_model
    - text_model
    - code_predictor_model
- code2wav_model

适配todo:
- 模型加载即权重精度转换
- thinker forward
- talker forward and code2wav

"""

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.thinker = Thinker(config.thinker_config)
        self.talker = Talker(config.talker_config)  # lazy
        self.code2wav = Code2WavModel(config.code2wav_config)  # lazy

        # if config.enable_audio_output:
        #     self.talker = None
        #     self.code2wav = None

    def get_input_embeddings(
        self, 
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        **kwargs,  # audio_features, image_grid_thw in kwargs.  
    ):
        pass

    def get_audio_features(
        self, 
        input_features, 
        input_features_mask):
        pass

    def get_image_features(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None
    ):
        pass
                
    @property
    def layers(self):
        return self.thinker.language_model.layers

    # def sanitize(self, weights: dict):
    #     sanitized_weights = {}
    #     for k, v in weights.items():
    #         if "thinker.model" in k:
    #             new_k = k.replace("model", "language_model.model")
    #             sanitized_weights[new_k] = v
    #         else:
    #             sanitized_weights[k] = v
    #     return sanitized_weights



    def __call__(
        self, 
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,  # audio_features in kwargs.
    ):
        pass
    