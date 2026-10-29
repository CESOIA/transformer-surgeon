"""Family dictionary of the modernbert_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_modernbert_c import MODERNBERT_C_INDEXING

MODERNBERT_C_FAMILY = {
    "indexing": MODERNBERT_C_INDEXING,
    "hf_config": "transformers.models.modernbert.configuration_modernbert.ModernBertConfig",
    "hf_model": "transformers.models.modernbert.modeling_modernbert.ModernBertForSequenceClassification",
    "class_names": {
        "config": "ModernBertConfigCompress",
        "model": "ModernBertForSequenceClassificationCompress",
        "manager": "ModernBertCompressionSchemesManager",
    },
    "auto_class": "AutoModelForSequenceClassification",
}

_classes = build_family(MODERNBERT_C_FAMILY, __name__)
ModernBertConfigCompress = _classes["ModernBertConfigCompress"]
ModernBertForSequenceClassificationCompress = _classes["ModernBertForSequenceClassificationCompress"]
ModernBertCompressionSchemesManager = _classes["ModernBertCompressionSchemesManager"]

__all__ = ["MODERNBERT_C_FAMILY", "ModernBertConfigCompress", "ModernBertForSequenceClassificationCompress", "ModernBertCompressionSchemesManager"]
