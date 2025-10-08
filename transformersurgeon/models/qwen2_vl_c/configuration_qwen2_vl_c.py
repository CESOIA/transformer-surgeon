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

from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLVisionConfig,
    Qwen2VLTextConfig,
    Qwen2VLConfig,
)

from transformers.utils import logging

from ...utils.configuration import init_compression_config

logger = logging.get_logger(__name__)

from .indexing_qwen2_vl_c import QWEN2_VL_C_INDEXING as INDEXING

class Qwen2VLVisionConfigCompress(Qwen2VLVisionConfig):
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

class Qwen2VLTextConfigCompress(Qwen2VLTextConfig):
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

class Qwen2VLConfigCompress(Qwen2VLConfig):
    """
    Main configuration class for Qwen2VL with compression support.
    """
    
    sub_configs = {
        "vision_config": Qwen2VLVisionConfigCompress, 
        "text_config": Qwen2VLTextConfigCompress
    }

__all__ = ["Qwen2VLConfigCompress", "Qwen2VLVisionConfigCompress", "Qwen2VLTextConfigCompress"]