"""
Model definitions for DistilBERT model compression.
"""

from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)
from .configuration_distilbert_c import DistilBertConfigCompress
from .indexing_distilbert_c import DISTILBERT_C_INDEXING as INDEXING
from ...utils import replace_layers_upon_init


class DistilBertForSequenceClassificationCompress(
    DistilBertForSequenceClassification
):
    config_class = DistilBertConfigCompress

    def __init__(self, config: DistilBertConfigCompress):
        super().__init__(config)

        replace_layers_upon_init(
            self,
            INDEXING["distilbert"],
            config,
        )


__all__ = ["DistilBertForSequenceClassificationCompress"]
