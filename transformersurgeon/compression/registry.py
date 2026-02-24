from .pruning import (
    Pruner,
    validate_pruning_ratio,
    validate_pruning_mode,
    validate_pruning_criterion,
)
from .lrd import (
    LRDer,
    validate_lrd_rank
)
from .quantization import (
    Quantizer,
    validate_precision,
    validate_sparsity,
    validate_sparse_criterion,
    validate_sparse_reverse
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
    },
    "lrd": {
        "rank": dict(default="full", validator=validate_lrd_rank)
    },
    "quantization": {
        "precision": dict(default="full", validator=validate_precision),
        "sparsity": dict(default=0.0, validator=validate_sparsity),
        "sparse_criterion": dict(default="magnitude", validator=validate_sparse_criterion),
        "sparse_reverse": dict(default=False, validator=validate_sparse_reverse)
    }
}

__all__ = [
    "COMPRESSION_REGISTRY",
    "COMPRESSOR_DICT"
]