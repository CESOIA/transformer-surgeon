from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig
from . import BERT_C_INDEXING as INDEXING
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init,
    CompressionSchemesManager,
)

# Define configuration
class BertConfigCompress(BertConfig):
    def __init__(self, compression=None, **kwargs):
        super().__init__(**kwargs)
        init_compression_config(
            config_instance=self,
            indexing=INDEXING["bert"],
            **(compression or {})
        )

# Define model
class BertForSequenceClassificationCompress(BertForSequenceClassification):
    config_class = BertConfigCompress
    indexing = INDEXING
    def __init__(self, config: BertConfigCompress):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["bert"], config)

# Define compression manager
class BertCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):           
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "BertConfigCompress",
    "BertForSequenceClassificationCompress",
    "BertCompressionSchemesManager",
]