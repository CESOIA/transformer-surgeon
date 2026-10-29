"""Family dictionary of the qwen2_5_vl_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_qwen2_5_vl_c import QWEN2_5_VL_C_INDEXING

QWEN2_5_VL_C_FAMILY = {
    "indexing": QWEN2_5_VL_C_INDEXING,
    "hf_config": "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig",
    "hf_model": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
    "sub_configs": {
        "vision_config": "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLVisionConfig",
        "text_config": "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLTextConfig",
    },
    "class_names": {
        "config": "Qwen2_5_VLConfigCompress",
        "vision_config": "Qwen2_5_VLVisionConfigCompress",
        "text_config": "Qwen2_5_VLTextConfigCompress",
        "model": "Qwen2_5_VLForConditionalGenerationCompress",
        "manager": "Qwen2_5_VLCompressionSchemesManager",
    },
    "auto_class": "AutoModelForConditionalGeneration",
}

_classes = build_family(QWEN2_5_VL_C_FAMILY, __name__)
Qwen2_5_VLVisionConfigCompress = _classes["Qwen2_5_VLVisionConfigCompress"]
Qwen2_5_VLTextConfigCompress = _classes["Qwen2_5_VLTextConfigCompress"]
Qwen2_5_VLConfigCompress = _classes["Qwen2_5_VLConfigCompress"]
Qwen2_5_VLForConditionalGenerationCompress = _classes["Qwen2_5_VLForConditionalGenerationCompress"]
Qwen2_5_VLCompressionSchemesManager = _classes["Qwen2_5_VLCompressionSchemesManager"]

__all__ = ["QWEN2_5_VL_C_FAMILY", "Qwen2_5_VLVisionConfigCompress", "Qwen2_5_VLTextConfigCompress", "Qwen2_5_VLConfigCompress", "Qwen2_5_VLForConditionalGenerationCompress", "Qwen2_5_VLCompressionSchemesManager"]
