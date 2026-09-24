from .magnitude import *
from .gradient import *
from .random import *
from .mask_generation import (
    build_structured_mask,
    effective_num_pruned,
    effective_out_features,
    reduce_scores,
)
from .pattern_ops import (
    reduce_pattern_scores,
    reduce_member_to_patterns,
    tile_pattern_mask,
)

# Score-only functions (decoupled from mask generation). Gradient additionally
# consumes weight_grad, handled explicitly by the caller.
SCORE_FUNCTIONS = {
    "magnitude": score_magnitude,
    "gradient": score_gradient,
    "random": score_random,
}

__all__ = [
    "SCORE_FUNCTIONS",
    "build_structured_mask",
    "effective_num_pruned",
    "effective_out_features",
    "reduce_scores",
    "reduce_pattern_scores",
    "reduce_member_to_patterns",
    "tile_pattern_mask",
]
