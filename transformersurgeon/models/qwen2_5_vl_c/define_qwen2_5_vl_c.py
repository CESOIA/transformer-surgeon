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
    def __init__(self, **kwargs):
        # Set the sub-config classes to the standard versions before calling the parent constructor
        self.sub_configs["vision_config"] = Qwen2_5_VLVisionConfig
        self.sub_configs["text_config"] = Qwen2_5_VLTextConfig
        # Call the parent constructor to initialize standard config attributes
        super().__init__(**kwargs)
        # Update vision and text sub-configs to their compressed versions
        self.sub_configs["vision_config"] = Qwen2_5_VLVisionConfigCompress
        self.sub_configs["text_config"] = Qwen2_5_VLTextConfigCompress
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
class Qwen2_5_VLForConditionalGenerationCompress(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLConfigCompress
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
