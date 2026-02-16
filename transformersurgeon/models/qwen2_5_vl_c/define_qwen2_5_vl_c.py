from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLConfig,
)
from . import QWEN2_5_VL_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init, 
    CompressionSchemesManager,
)

# Define configuration
class Qwen2_5_VLVisionConfigCompress(Qwen2_5_VLVisionConfig):
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["vision"],
            **(compression or {})
        )
class Qwen2_5_VLTextConfigCompress(Qwen2_5_VLTextConfig):
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["text"],
            **(compression or {})
        )

class Qwen2_5_VLConfigCompress(Qwen2_5_VLConfig):
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfigCompress, 
        "text_config": Qwen2_5_VLTextConfigCompress
    }

# Define model
class Qwen2_5_VLForConditionalGenerationCompress(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLConfigCompress
    def __init__(self, config):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["vision"], config)
        replace_layers_upon_init(self, INDEXING["text"], config)

# Define compression manager
class Qwen2_5_VL_CompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "Qwen2_5_VLForConditionalGenerationCompress",
    "Qwen2_5_VLVisionConfigCompress", 
    "Qwen2_5_VLTextConfigCompress", 
    "Qwen2_5_VLConfigCompress",
    "Qwen2_5_VL_CompressionSchemesManager",
]
