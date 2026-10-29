"""Family dictionary of the mistral_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_mistral_c import MISTRAL_C_INDEXING

MISTRAL_C_FAMILY = {
    "indexing": MISTRAL_C_INDEXING,
    "hf_config": "transformers.models.mistral.configuration_mistral.MistralConfig",
    "hf_model": "transformers.models.mistral.modeling_mistral.MistralForCausalLM",
    "class_names": {
        "config": "MistralConfigCompress",
        "model": "MistralForCausalLMCompress",
        "manager": "MistralCompressionSchemesManager",
    },
    "auto_class": "AutoModelForCausalLM",
}

_classes = build_family(MISTRAL_C_FAMILY, __name__)
MistralConfigCompress = _classes["MistralConfigCompress"]
MistralForCausalLMCompress = _classes["MistralForCausalLMCompress"]
MistralCompressionSchemesManager = _classes["MistralCompressionSchemesManager"]

__all__ = ["MISTRAL_C_FAMILY", "MistralConfigCompress", "MistralForCausalLMCompress", "MistralCompressionSchemesManager"]
