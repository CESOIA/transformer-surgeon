import inspect
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
    def __init__(self, **kwargs):
        compression_config = kwargs.pop("compression_config", {})
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["vision"],
            compression_config=compression_config
        )
class Qwen2_5_VLTextConfigCompress(Qwen2_5_VLTextConfig):
    def __init__(self, **kwargs):
        compression_config = kwargs.pop("compression_config", {})
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["text"],
            compression_config=compression_config
        )

class Qwen2_5_VLConfigCompress(Qwen2_5_VLConfig):
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfigCompress,
        "text_config": Qwen2_5_VLTextConfigCompress,
    }
    # This is needed to make sure the signature of the compressed config matches the original one for Hugging Face compatibility
    Qwen2_5_VLTextConfigCompress.__init__.__signature__ = inspect.signature(Qwen2_5_VLTextConfig.__init__)

# Define model
class Qwen2_5_VLForConditionalGenerationCompress(Qwen2_5_VLForConditionalGeneration):
    config: Qwen2_5_VLConfigCompress
    indexing = INDEXING
    def __init__(self, config): 
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["vision"], config)
        replace_layers_upon_init(self, INDEXING["text"], config)

# Define compression manager
class Qwen2_5_VLCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "Qwen2_5_VLForConditionalGenerationCompress",
    "Qwen2_5_VLVisionConfigCompress", 
    "Qwen2_5_VLTextConfigCompress", 
    "Qwen2_5_VLConfigCompress",
    "Qwen2_5_VLCompressionSchemesManager",
]
