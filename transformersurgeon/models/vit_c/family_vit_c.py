"""Family dictionary of the vit_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_vit_c import VIT_C_INDEXING
from .legacy_conversion import legacy_checkpoint_key_mapping

VIT_C_FAMILY = {
    "indexing": VIT_C_INDEXING,
    "hf_config": "transformers.models.vit.configuration_vit.ViTConfig",
    "hf_model": "transformers.models.vit.modeling_vit.ViTForImageClassification",
    "class_names": {
        "config": "ViTConfigCompress",
        "model": "ViTForImageClassificationCompress",
        "manager": "ViTCompressionSchemesManager",
    },
    "auto_class": "AutoModelForImageClassification",
    "checkpoint_key_mapping": legacy_checkpoint_key_mapping,
}

_classes = build_family(VIT_C_FAMILY, __name__)
ViTConfigCompress = _classes["ViTConfigCompress"]
ViTForImageClassificationCompress = _classes["ViTForImageClassificationCompress"]
ViTCompressionSchemesManager = _classes["ViTCompressionSchemesManager"]

__all__ = ["VIT_C_FAMILY", "ViTConfigCompress", "ViTForImageClassificationCompress", "ViTCompressionSchemesManager"]
