import inspect
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from . import QWEN2_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init, 
    CompressionSchemesManager,
)

# Define configuration
class Qwen2ConfigCompress(Qwen2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["text"],
            compression_config=kwargs.get("compression_config", {})
        )

# Define model
class Qwen2ForCausalLMCompress(Qwen2ForCausalLM):
    config_class = Qwen2ConfigCompress
    indexing = INDEXING
    def __init__(self, config):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["text"], config)

# Define compression manager
class Qwen2CompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "Qwen2ConfigCompress",
    "Qwen2ForCausalLMCompress",
    "Qwen2CompressionSchemesManager",
]
