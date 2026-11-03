"""Family dictionary of the dinov2_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_dinov2_c import DINOV2_C_INDEXING, DINOV2_SWIGLU_C_INDEXING

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

# Checkpoints with use_swiglu_ffn=True (DINOv2-giant): same Hugging Face classes,
# model dictionary with the fused SwiGLU MLP.
DINOV2_SWIGLU_C_FAMILY = {
    **DINOV2_C_FAMILY,
    "indexing": DINOV2_SWIGLU_C_INDEXING,
    "class_names": {
        "config": "Dinov2SwiGLUConfigCompress",
        "model": "Dinov2SwiGLUForImageClassificationCompress",
        "manager": "Dinov2SwiGLUCompressionSchemesManager",
    },
}

_classes = build_family(DINOV2_SWIGLU_C_FAMILY, __name__)
Dinov2SwiGLUConfigCompress = _classes["Dinov2SwiGLUConfigCompress"]
Dinov2SwiGLUForImageClassificationCompress = _classes["Dinov2SwiGLUForImageClassificationCompress"]
Dinov2SwiGLUCompressionSchemesManager = _classes["Dinov2SwiGLUCompressionSchemesManager"]

__all__ = ["DINOV2_C_FAMILY", "Dinov2ConfigCompress", "Dinov2ForImageClassificationCompress",
           "Dinov2CompressionSchemesManager", "DINOV2_SWIGLU_C_FAMILY", "Dinov2SwiGLUConfigCompress",
           "Dinov2SwiGLUForImageClassificationCompress", "Dinov2SwiGLUCompressionSchemesManager"]
