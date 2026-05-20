from .structured_pruning import (
    StructuredPruner,
    validate_structured_pruning_ratio,
    validate_structured_pruning_method,
)
from .unstructured_pruning import (
    UnstructuredPruner,
    validate_unstructured_pruning_ratio,
    validate_unstructured_pruning_method,
    validate_unstructured_pruning_granularity,
)
from .lrd import (
    LRDer,
    validate_lrd_rank,
    validate_lrd_method,
    validate_lrd_eps,
)
from .quantization import (
    Quantizer,
    validate_quantization_method,
    validate_precision,
    validate_sparse_method,
    validate_quantization_eps,
)

COMPRESSOR_DICT = {
    "structured_pruning": StructuredPruner,
    "unstructured_pruning": UnstructuredPruner,
    "lrd": LRDer,
    "quantization": Quantizer
}

COMPRESSION_REGISTRY = {
    "structured_pruning": {
        "ratio": dict(default=0.0, validator=validate_structured_pruning_ratio),
        "method": dict(default="magnitude", validator=validate_structured_pruning_method),
    },
    "unstructured_pruning": {
        "ratio": dict(default=0.0, validator=validate_unstructured_pruning_ratio),
        "method": dict(default="magnitude", validator=validate_unstructured_pruning_method),
        "granularity": dict(default="layer", validator=validate_unstructured_pruning_granularity),
    },
    "lrd": {
        "rank": dict(default="full", validator=validate_lrd_rank),
        "method": dict(default="svd", validator=validate_lrd_method),
        "eps": dict(default=1e-6, validator=validate_lrd_eps),
    },
    "quantization": {
        "method": dict(default="vanilla", validator=validate_quantization_method),
        "precision": dict(default="full", validator=validate_precision),
        "sparsity": dict(default=0.0, validator=validate_unstructured_pruning_ratio),
        "sparse_method": dict(default="magnitude", validator=validate_sparse_method,),
        "eps": dict(default=1e-6, validator=validate_quantization_eps),
    }
}

__all__ = [
    "COMPRESSION_REGISTRY",
    "COMPRESSOR_DICT",
]
