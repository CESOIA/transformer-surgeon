from transformers.models.modernbert.modeling_modernbert import ModernBertForSequenceClassification
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from . import MODERNBERT_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init,
    CompressionSchemesManager,
)

# Define configuration
class ModernBertConfigCompress(ModernBertConfig):
    def __init__(self, compression=None, **kwargs):
        compression_config = kwargs.pop("compression_config", {})
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["modernbert"],
            compression_config=compression_config
        )

# Define model
class ModernBertForSequenceClassificationCompress(ModernBertForSequenceClassification):
    config_class = ModernBertConfigCompress
    indexing = INDEXING
    def __init__(self, config: ModernBertConfigCompress):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["modernbert"], config)

# Define compression manager
class ModernBertCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "ModernBertConfigCompress",
    "ModernBertForSequenceClassificationCompress",
    "ModernBertCompressionSchemesManager",
]
