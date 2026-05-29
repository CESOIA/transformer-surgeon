from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from . import LLAMA_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init,
    CompressionSchemesManager,
)


# Define configuration
class LlamaConfigCompress(LlamaConfig):
    def __init__(self, **kwargs):
        compression_config = kwargs.pop("compression_config", {})
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["text"],
            compression_config=compression_config,
        )


# Define model
class LlamaForCausalLMCompress(LlamaForCausalLM):
    config_class = LlamaConfigCompress
    indexing = INDEXING

    def __init__(self, config):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["text"], config)


# Define compression manager
class LlamaCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)


__all__ = [
    "LlamaConfigCompress",
    "LlamaForCausalLMCompress",
    "LlamaCompressionSchemesManager",
]
