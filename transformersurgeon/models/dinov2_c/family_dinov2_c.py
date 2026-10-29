"""Family dictionary of the dinov2_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_dinov2_c import DINOV2_C_INDEXING

DINOV2_C_FAMILY = {
    "indexing": DINOV2_C_INDEXING,
    "hf_config": "transformers.models.dinov2.configuration_dinov2.Dinov2Config",
    "hf_model": "transformers.models.dinov2.modeling_dinov2.Dinov2ForImageClassification",
    "class_names": {
        "config": "Dinov2ConfigCompress",
        "model": "Dinov2ForImageClassificationCompress",
        "manager": "Dinov2CompressionSchemesManager",
    },
    "auto_class": "AutoModelForImageClassification",
}

_classes = build_family(DINOV2_C_FAMILY, __name__)
Dinov2ConfigCompress = _classes["Dinov2ConfigCompress"]
Dinov2ForImageClassificationCompress = _classes["Dinov2ForImageClassificationCompress"]
Dinov2CompressionSchemesManager = _classes["Dinov2CompressionSchemesManager"]

__all__ = ["DINOV2_C_FAMILY", "Dinov2ConfigCompress", "Dinov2ForImageClassificationCompress",
           "Dinov2CompressionSchemesManager"]
