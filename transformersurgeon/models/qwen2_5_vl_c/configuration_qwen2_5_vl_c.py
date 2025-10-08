# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLConfig,
)

from ...utils.configuration import init_compression_config

from .indexing_qwen2_5_vl_c import QWEN2_5_VL_C_INDEXING as INDEXING

class Qwen2_5_VLVisionConfigCompress(Qwen2_5_VLVisionConfig):
    """Vision configuration with compression support."""
    
    def __init__(
        self,
        pruning_ratios=None,
        lrd_ranks=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        init_compression_config(
            config_instance=self,
            indexing=INDEXING["vision"],
            pruning_ratios=pruning_ratios,
            lrd_ranks=lrd_ranks,
        )

class Qwen2_5_VLTextConfigCompress(Qwen2_5_VLTextConfig):
    """Text configuration with compression support."""
    
    def __init__(
        self,
        pruning_ratios=None,
        lrd_ranks=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        init_compression_config(
            config_instance=self,
            indexing=INDEXING["text"],
            pruning_ratios=pruning_ratios,
            lrd_ranks=lrd_ranks,
        )

class Qwen2_5_VLConfigCompress(Qwen2_5_VLConfig):
    """
    Main configuration class for Qwen2.5VL with compression support.
    """
    
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfigCompress, 
        "text_config": Qwen2_5_VLTextConfigCompress
    }

__all__ = [
    "Qwen2_5_VLVisionConfigCompress", 
    "Qwen2_5_VLTextConfigCompress", 
    "Qwen2_5_VLConfigCompress"
]