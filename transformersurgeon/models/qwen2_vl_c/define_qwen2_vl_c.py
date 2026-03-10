import inspect
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLVisionConfig,
    Qwen2VLTextConfig,
    Qwen2VLConfig,
)
from . import QWEN2_VL_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init, 
    CompressionSchemesManager,
)

# Define configuration
class Qwen2VLVisionConfigCompress(Qwen2VLVisionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["vision"],
            compression_config=kwargs.get("compression_config", {})
        )
class Qwen2VLTextConfigCompress(Qwen2VLTextConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["text"],
            compression_config=kwargs.get("compression_config", {})
        )

class Qwen2VLConfigCompress(Qwen2VLConfig):
    sub_configs = {
        "vision_config": Qwen2VLVisionConfigCompress,
        "text_config": Qwen2VLTextConfigCompress,
    }
    # This is needed to make sure the signature of the compressed config matches the original one for Hugging Face compatibility
    Qwen2VLTextConfigCompress.__init__.__signature__ = inspect.signature(Qwen2VLTextConfig.__init__)

# Define model
class Qwen2VLForConditionalGenerationCompress(Qwen2VLForConditionalGeneration):
    config_class = Qwen2VLConfigCompress
    indexing = INDEXING
    def __init__(self, config):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["vision"], config)
        replace_layers_upon_init(self, INDEXING["text"], config)

# Define compression manager
class Qwen2VLCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "Qwen2VLForConditionalGenerationCompress",
    "Qwen2VLVisionConfigCompress", 
    "Qwen2VLTextConfigCompress", 
    "Qwen2VLConfigCompress",
    "Qwen2VLCompressionSchemesManager",
]
