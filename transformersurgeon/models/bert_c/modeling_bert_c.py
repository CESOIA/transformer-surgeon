"""
Model definitions for Bert model compression.
"""

from transformers.models.bert.modeling_bert import BertForSequenceClassification
from .configuration_bert_c import BertConfigCompress
from .indexing_bert_c import BERT_C_INDEXING as INDEXING
from ...utils import replace_layers_upon_init

class BertForSequenceClassificationCompress(BertForSequenceClassification):
    config_class = BertConfigCompress

    def __init__(self, config: BertConfigCompress):
        super().__init__(config)

        replace_layers_upon_init(
            self,
            INDEXING["bert"],
            config,
        )

__all__ = ["BertForSequenceClassificationCompress"]