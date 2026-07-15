from .structured_pruning import (
    StructuredPruner,
    validate_structured_pruning_ratio,
    validate_structured_pruning_method,
    validate_structured_pruning_granularity,
    validate_structured_pruning_share_mask,
    validate_structured_pruning_reduce_op,
    validate_structured_pruning_repeated_pattern,
    validate_structured_pruning_coupled_repeated_pattern,
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
    validate_activation_method,
    validate_precision,
    validate_sparse_method,
    validate_quantization_eps,
    validate_quantization_granularity,
    validate_activation_scheme,
)

COMPRESSOR_DICT = {
    "structured_pruning": StructuredPruner,
    "unstructured_pruning": UnstructuredPruner,
    "lrd": LRDer,
    "quantization": Quantizer
}

# Canonical order in which compressors are applied to a module.
# lrd first (changes weight shape), then pruning (masks the result),
# then quantization (must see the final float weights).
APPLICATION_ORDER = ["lrd", "structured_pruning", "unstructured_pruning", "quantization"]

# Compression properties that are group-scoped: they may be set ONLY through a
# group name (manager.set(..., group=...)), never via plain criteria. Enabling
# any of them resets the affected scheme's (non-group) compression config to
# defaults first. The list is keyed by compression type. reduce_op is NOT here:
# it is also used per-layer by repeated_pattern, so it must be settable via
# criteria too.
GROUP_OPTIONS = {
    "structured_pruning": ["share_mask"],
}

COMPRESSION_REGISTRY = {
    "structured_pruning": {
        "ratio": dict(default=0.0, validator=validate_structured_pruning_ratio),
        "method": dict(default="magnitude", validator=validate_structured_pruning_method),
        "granularity": dict(default="layer", validator=validate_structured_pruning_granularity),
        # repeated_pattern: prune the same position(s) within groups of size
        # granularity (e.g. attention heads). True/"max" reduces all groups into
        # one shared length-granularity mask (lets layers with different group
        # counts share one mask, e.g. GQA q/k); a positive int N instead
        # re-derives an independent length-granularity mask every N consecutive
        # groups and repeats it only across that run. "auto" (shared-mask groups
        # only) derives one pattern per kv-group from the group's shapes -- the
        # pattern count = the smallest member's group count (= num_kv_heads) --
        # and each member tiles it by its own num_groups // pattern_count.
        "repeated_pattern": dict(default=False, validator=validate_structured_pruning_repeated_pattern),
        # coupled_repeated_pattern: when cascading this layer's keep-mask onto
        # coupled next layers (hard apply), repeat each length-granularity
        # chunk of the mask N times in place (chunk chunk ... | next_chunk
        # next_chunk ...) instead of forwarding it 1:1. For a coupled next
        # layer whose input dimension is a repeated expansion of this layer's
        # own pruned output dimension.
        "coupled_repeated_pattern": dict(
            default=False, validator=validate_structured_pruning_coupled_repeated_pattern
        ),
        # reduce_op: how scores are combined — across granularity groups for
        # repeated_pattern, and/or across layers for a shared-mask group.
        "reduce_op": dict(default=None, validator=validate_structured_pruning_reduce_op),
        # Group-only option (see GROUP_OPTIONS): settable only via a group name.
        "share_mask": dict(default=False, validator=validate_structured_pruning_share_mask),
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
        "granularity": dict(default="per_tensor", validator=validate_quantization_granularity),
        "precision_activation": dict(default="full", validator=validate_precision),
        "method_activation": dict(default="maxmin", validator=validate_activation_method),
        "scheme_activation": dict(default="asymmetric", validator=validate_activation_scheme),
    }
}

__all__ = [
    "COMPRESSION_REGISTRY",
    "COMPRESSOR_DICT",
    "APPLICATION_ORDER",
    "GROUP_OPTIONS",
]
