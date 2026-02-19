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
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["vision"],
            **(compression or {})
        )
class Qwen2VLTextConfigCompress(Qwen2VLTextConfig):
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["text"],
            **(compression or {})
        )

class Qwen2VLConfigCompress(Qwen2VLConfig):
    def __init__(self, **kwargs):
        # Set the sub-config classes to the standard versions before calling the parent constructor
        self.sub_configs["vision_config"] = Qwen2VLVisionConfig
        self.sub_configs["text_config"] = Qwen2VLTextConfig
        # Call the parent constructor to initialize standard config attributes
        super().__init__(**kwargs)
        # Update vision and text sub-configs to their compressed versions
        self.sub_configs["vision_config"] = Qwen2VLVisionConfigCompress
        self.sub_configs["text_config"] = Qwen2VLTextConfigCompress
        # Convert the vision and text sub-configs to their compressed versions, passing the compression kwargs
        vision_config_dict = {} if self.vision_config is None else self.vision_config.to_dict()
        text_config_dict = {} if self.text_config is None else self.text_config.to_dict()
        self.vision_config = self.sub_configs["vision_config"](
            compression=kwargs.get("vision_config", {}).get("compression_config", None),
            **vision_config_dict)
        self.text_config = self.sub_configs["text_config"](
            compression=kwargs.get("text_config", {}).get("compression_config", None),
            **text_config_dict)

# Define model
class Qwen2VLForConditionalGenerationCompress(Qwen2VLForConditionalGeneration):
    config_class = Qwen2VLConfigCompress
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
