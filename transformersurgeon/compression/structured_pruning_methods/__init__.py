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

# Legacy score+mask helpers (kept for backward compatibility).
METHOD_FUNCTIONS = {
    "magnitude": mask_magnitude,
    "gradient": mask_gradient,
    "random": mask_random,
}

__all__ = [
    "SCORE_FUNCTIONS",
    "METHOD_FUNCTIONS",
    "build_structured_mask",
    "effective_num_pruned",
    "effective_out_features",
    "reduce_scores",
    "reduce_pattern_scores",
    "reduce_member_to_patterns",
    "tile_pattern_mask",
]
