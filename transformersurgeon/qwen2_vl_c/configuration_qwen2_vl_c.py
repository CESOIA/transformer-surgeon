# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen2VL model configuration - patched for compression algorithms."""

from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLVisionConfig,
    Qwen2VLTextConfig,
    Qwen2VLConfig,
)
from transformers.utils import logging

from ..utils.configuration import init_compression_config

logger = logging.get_logger(__name__)

### -----------------------------------
### COMPRESSION CONFIG CLASSES
### -----------------------------------

class Qwen2VLVisionConfigCompress(Qwen2VLVisionConfig):
    """Vision configuration with compression support."""
    
    def __init__(
        self,
        pruning_ratio_lists=None,
        pruning_ratio_skip_connections=None,
        lrd_rank_lists=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define default pruning and lrd keys
        default_pruning_keys = ["mlp_up"]
        default_lrd_keys = ["sa_qkv", "sa_out", "mlp_up", "mlp_down"]
        
        # Define layer type mappings for vision model - separate for pruning and LRD
        pruning_key_mappings = {
            "all": ["mlp_up"],
            # "sa":  ["sa_qkv"],
            "mlp": ["mlp_up"]
        }
        
        lrd_key_mappings = {
            "all": ["sa_qkv", "sa_out", "mlp_up", "mlp_down"],
            "sa":  ["sa_qkv", "sa_out"],
            "mlp": ["mlp_up", "mlp_down"]
        }
        
        init_compression_config(
            config_instance=self,
            total_blocks=self.depth,
            base_dim=self.embed_dim,
            pruning_ratio_lists=pruning_ratio_lists,
            pruning_ratio_skip_connections=pruning_ratio_skip_connections,
            lrd_rank_lists=lrd_rank_lists,
            default_pruning_keys=default_pruning_keys,
            default_lrd_keys=default_lrd_keys,
            pruning_key_mappings=pruning_key_mappings,
            lrd_key_mappings=lrd_key_mappings,
            mlp_ratio=self.mlp_ratio,
            num_heads=self.num_heads
        )

class Qwen2VLTextConfigCompress(Qwen2VLTextConfig):
    """Text configuration with compression support."""
    
    def __init__(
        self,
        pruning_ratio_lists=None,
        pruning_ratio_skip_connections=None,
        lrd_rank_lists=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define default pruning and lrd keys
        # default_pruning_keys = ["mlp_gate", "mlp_up"]
        default_pruning_keys = []
        default_lrd_keys = ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"]
        
        # Define layer type mappings for text model - separate for pruning and LRD
        pruning_key_mappings = {
            # "all":    ["mlp_gate", "mlp_up"],
            # "sa":     ["sa_q", "sa_k", "sa_v"],
            # "sa_qkv": ["sa_q", "sa_k", "sa_v"],
            # "mlp":    ["mlp_gate", "mlp_up"]
        }
        
        lrd_key_mappings = {
            "all":    ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"],
            "sa":     ["sa_q", "sa_k", "sa_v", "sa_out"],
            "sa_qkv": ["sa_q", "sa_k", "sa_v"],
            "mlp":    ["mlp_gate", "mlp_up", "mlp_down"]
        }
        
        init_compression_config(
            config_instance=self,
            total_blocks=self.num_hidden_layers,
            base_dim=self.hidden_size,
            pruning_ratio_lists=pruning_ratio_lists,
            pruning_ratio_skip_connections=pruning_ratio_skip_connections,
            lrd_rank_lists=lrd_rank_lists,
            default_pruning_keys=default_pruning_keys,
            default_lrd_keys=default_lrd_keys,
            pruning_key_mappings=pruning_key_mappings,
            lrd_key_mappings=lrd_key_mappings,
            num_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size
        )

class Qwen2VLConfigCompress(Qwen2VLConfig):
    """
    Main configuration class for Qwen2VL with compression support.
    
    This configuration extends the original Qwen2VLConfig with compression capabilities
    including structured pruning and low-rank decomposition (LRD).
    """
    
    model_type = "qwen2_vl"
    sub_configs = {
        "vision_config": Qwen2VLVisionConfigCompress, 
        "text_config": Qwen2VLTextConfigCompress
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        **kwargs,
    ):
        # Handle vision config
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        # Handle text config
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['pruning_ratio_lists', 'pruning_ratio_skip_connections', 'lrd_rank_lists']}
            self.text_config = self.sub_configs["text_config"](**filtered_kwargs)
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        parent_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['pruning_ratio_lists', 'pruning_ratio_skip_connections', 'lrd_rank_lists',
                                   'text_config', 'vision_config']}
        
        super(Qwen2VLConfig, self).__init__(**parent_kwargs)

__all__ = ["Qwen2VLConfigCompress", "Qwen2VLVisionConfigCompress", "Qwen2VLTextConfigCompress"]