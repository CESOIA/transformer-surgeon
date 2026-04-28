from .pruning import (
    Pruner,
    validate_pruning_ratio,
    validate_pruning_mode,
    validate_pruning_criterion,
    validate_pruning_granularity
)
from .lrd import (
    LRDer,
    validate_lrd_method,
    validate_lrd_rank
)
from .quantization import (
    Quantizer,
    validate_precision,
    validate_sparsity,
)

COMPRESSOR_DICT = {
    "pruning": Pruner,
    "lrd": LRDer,
    "quantization": Quantizer
}

COMPRESSION_REGISTRY = {
    "pruning": {
        "ratio": dict(default=0.0, validator=validate_pruning_ratio),
        "mode": dict(default="structured", validator=validate_pruning_mode),
        "criterion": dict(default="magnitude", validator=validate_pruning_criterion),
        "granularity": dict(default="layer", validator=validate_pruning_granularity)
    },
    "lrd": {
        "rank": dict(default="full", validator=validate_lrd_rank),
        "method": dict(default="svd", validator=validate_lrd_method)
    },
    "quantization": {
        "precision": dict(default="full", validator=validate_precision),
        "sparsity": dict(default=0.0, validator=validate_sparsity),
        "sparse_criterion": dict(default="magnitude", validator=validate_pruning_criterion,),
    }
}

__all__ = [
    "COMPRESSION_REGISTRY",
    "COMPRESSOR_DICT"
]