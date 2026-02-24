from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from . import DISTILBERT_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init,
    CompressionSchemesManager,
)

# Define configuration
class DistilBertConfigCompress(DistilBertConfig):
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compression_config(
            config_instance=self,
            indexing=INDEXING["distilbert"],
            **(compression or {})
        )

# Define model
class DistilBertForSequenceClassificationCompress(DistilBertForSequenceClassification):
    config_class = DistilBertConfigCompress
    indexing = INDEXING
    def __init__(self, config: DistilBertConfigCompress):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["distilbert"], config)

# Define compression manager
class DistilBertCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

__all__ = [
    "DistilBertConfigCompress",
    "DistilBertForSequenceClassificationCompress",
    "DistilBertCompressionSchemesManager",
]