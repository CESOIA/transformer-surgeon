"""
Model definitions for ViT model compression.
"""

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .configuration_qwen2_5_vl_c import Qwen2_5_VLConfigCompress
from .indexing_qwen2_5_vl_c import QWEN2_5_VL_C_INDEXING as INDEXING
from ...utils import replace_layers_upon_init

class Qwen2_5_VLForConditionalGenerationCompress(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLConfigCompress

    def __init__(self, config: Qwen2_5_VLConfigCompress):
        super().__init__(config)

        replace_layers_upon_init(
            self,
            INDEXING["vision"],
            config,
        )

        replace_layers_upon_init(
            self,
            INDEXING["text"],
            config,
        )

__all__ = ["Qwen2_5_VLForConditionalGenerationCompress"]