"""
Model definitions for Qwen2-VL model compression.
"""

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from .configuration_qwen2_vl_c import Qwen2VLConfigCompress
from .indexing_qwen2_vl_c import QWEN2_VL_C_INDEXING as INDEXING
from ...utils import replace_layers_upon_init

class Qwen2VLForConditionalGenerationCompress(Qwen2VLForConditionalGeneration):
    config_class = Qwen2VLConfigCompress

    def __init__(self, config: Qwen2VLConfigCompress):
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

__all__ = ["Qwen2VLForConditionalGenerationCompress"]