"""Family dictionary of the qwen2_vl_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_qwen2_vl_c import QWEN2_VL_C_INDEXING

QWEN2_VL_C_FAMILY = {
    "indexing": QWEN2_VL_C_INDEXING,
    "hf_config": "transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLConfig",
    "hf_model": "transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration",
    "sub_configs": {
        "vision_config": "transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLVisionConfig",
        "text_config": "transformers.models.qwen2_vl.configuration_qwen2_vl.Qwen2VLTextConfig",
    },
    "class_names": {
        "config": "Qwen2VLConfigCompress",
        "vision_config": "Qwen2VLVisionConfigCompress",
        "text_config": "Qwen2VLTextConfigCompress",
        "model": "Qwen2VLForConditionalGenerationCompress",
        "manager": "Qwen2VLCompressionSchemesManager",
    },
    "auto_class": "AutoModelForConditionalGeneration",
}

_classes = build_family(QWEN2_VL_C_FAMILY, __name__)
Qwen2VLVisionConfigCompress = _classes["Qwen2VLVisionConfigCompress"]
Qwen2VLTextConfigCompress = _classes["Qwen2VLTextConfigCompress"]
Qwen2VLConfigCompress = _classes["Qwen2VLConfigCompress"]
Qwen2VLForConditionalGenerationCompress = _classes["Qwen2VLForConditionalGenerationCompress"]
Qwen2VLCompressionSchemesManager = _classes["Qwen2VLCompressionSchemesManager"]

__all__ = ["QWEN2_VL_C_FAMILY", "Qwen2VLVisionConfigCompress", "Qwen2VLTextConfigCompress", "Qwen2VLConfigCompress", "Qwen2VLForConditionalGenerationCompress", "Qwen2VLCompressionSchemesManager"]
