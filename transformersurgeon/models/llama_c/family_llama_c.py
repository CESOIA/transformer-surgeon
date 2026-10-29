"""Family dictionary of the llama_c compressible model classes (see ``models/factory.py``)."""

from ..factory import build_family
from .indexing_llama_c import LLAMA_C_INDEXING

LLAMA_C_FAMILY = {
    "indexing": LLAMA_C_INDEXING,
    "hf_config": "transformers.models.llama.configuration_llama.LlamaConfig",
    "hf_model": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
    "class_names": {
        "config": "LlamaConfigCompress",
        "model": "LlamaForCausalLMCompress",
        "manager": "LlamaCompressionSchemesManager",
    },
    "auto_class": "AutoModelForCausalLM",
}

_classes = build_family(LLAMA_C_FAMILY, __name__)
LlamaConfigCompress = _classes["LlamaConfigCompress"]
LlamaForCausalLMCompress = _classes["LlamaForCausalLMCompress"]
LlamaCompressionSchemesManager = _classes["LlamaCompressionSchemesManager"]

__all__ = ["LLAMA_C_FAMILY", "LlamaConfigCompress", "LlamaForCausalLMCompress", "LlamaCompressionSchemesManager"]
