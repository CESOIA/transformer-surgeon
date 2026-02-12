from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers.models.vit.configuration_vit import ViTConfig
from . import VIT_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init, 
    CompressionSchemesManager,
)

# Define configuration
class ViTConfigCompress(ViTConfig):
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["vit"],
            **(compression or {})
        )

# Define model
class ViTForImageClassificationCompress(ViTForImageClassification):
    config_class = ViTConfigCompress
    def __init__(self, config):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["vit"], config)

# Define compression manager
class ViTCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "ViTConfigCompress",
    "ViTForImageClassificationCompress",
    "ViTCompressionSchemesManager",
]
