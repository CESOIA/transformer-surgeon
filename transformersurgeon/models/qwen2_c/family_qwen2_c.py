"""Family dictionary of the qwen2_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_qwen2_c import QWEN2_C_INDEXING

QWEN2_C_FAMILY = {
    "indexing": QWEN2_C_INDEXING,
    "hf_config": "transformers.models.qwen2.configuration_qwen2.Qwen2Config",
    "hf_model": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
    "class_names": {
        "config": "Qwen2ConfigCompress",
        "model": "Qwen2ForCausalLMCompress",
        "manager": "Qwen2CompressionSchemesManager",
    },
    "auto_class": "AutoModelForCausalLM",
}

_classes = build_family(QWEN2_C_FAMILY, __name__)
Qwen2ConfigCompress = _classes["Qwen2ConfigCompress"]
Qwen2ForCausalLMCompress = _classes["Qwen2ForCausalLMCompress"]
Qwen2CompressionSchemesManager = _classes["Qwen2CompressionSchemesManager"]

__all__ = ["QWEN2_C_FAMILY", "Qwen2ConfigCompress", "Qwen2ForCausalLMCompress", "Qwen2CompressionSchemesManager"]
