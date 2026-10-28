"""Family dictionary of the distilbert_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_distilbert_c import DISTILBERT_C_INDEXING

DISTILBERT_C_FAMILY = {
    "indexing": DISTILBERT_C_INDEXING,
    "hf_config": "transformers.models.distilbert.configuration_distilbert.DistilBertConfig",
    "hf_model": "transformers.models.distilbert.modeling_distilbert.DistilBertForSequenceClassification",
    "class_names": {
        "config": "DistilBertConfigCompress",
        "model": "DistilBertForSequenceClassificationCompress",
        "manager": "DistilBertCompressionSchemesManager",
    },
    "auto_class": "AutoModelForSequenceClassification",
}

_classes = build_family(DISTILBERT_C_FAMILY, __name__)
DistilBertConfigCompress = _classes["DistilBertConfigCompress"]
DistilBertForSequenceClassificationCompress = _classes["DistilBertForSequenceClassificationCompress"]
DistilBertCompressionSchemesManager = _classes["DistilBertCompressionSchemesManager"]

__all__ = ["DISTILBERT_C_FAMILY", "DistilBertConfigCompress", "DistilBertForSequenceClassificationCompress", "DistilBertCompressionSchemesManager"]
