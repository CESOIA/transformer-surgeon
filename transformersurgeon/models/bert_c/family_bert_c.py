"""Family dictionary of the bert_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_bert_c import BERT_C_INDEXING

BERT_C_FAMILY = {
    "indexing": BERT_C_INDEXING,
    "hf_config": "transformers.models.bert.configuration_bert.BertConfig",
    "hf_model": "transformers.models.bert.modeling_bert.BertForSequenceClassification",
    "class_names": {
        "config": "BertConfigCompress",
        "model": "BertForSequenceClassificationCompress",
        "manager": "BertCompressionSchemesManager",
    },
    "auto_class": "AutoModelForSequenceClassification",
}

_classes = build_family(BERT_C_FAMILY, __name__)
BertConfigCompress = _classes["BertConfigCompress"]
BertForSequenceClassificationCompress = _classes["BertForSequenceClassificationCompress"]
BertCompressionSchemesManager = _classes["BertCompressionSchemesManager"]

__all__ = ["BERT_C_FAMILY", "BertConfigCompress", "BertForSequenceClassificationCompress", "BertCompressionSchemesManager"]
