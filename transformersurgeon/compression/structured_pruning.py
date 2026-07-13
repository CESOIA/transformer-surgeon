import logging
import torch
import torch.nn as nn

from .abstract import Compressor
from .coupled_pruning import CoupledPruner
from .structured_pruning_methods import (
    SCORE_FUNCTIONS,
    build_structured_mask,
    reduce_scores,
    reduce_pattern_scores,
    tile_pattern_mask,
)
from ..blocks import EmbeddingCompressed

logger = logging.getLogger(__name__)

# The model's preprocessing/embedding layer (nn.Embedding -> EmbeddingCompressed)
# and final layer (lm_head/classifier, nn.Linear -> LinearCompressed) are wired
# into hidden-dim structured pruning via the 'preprocessing'/'final_layer'
# sentinels in indexing's pruning.output_dependence / coupled_masks_all (see
# _coupled_target_chains below and manager._resolve_sentinel_path). Not every
# family supports this: models whose preprocessing module isn't a plain
# nn.Embedding (e.g. ViT/Qwen-VL patch-embed Conv2d, BERT-style composite
# embeddings) don't get a 'preprocessing' scheme, so only their 'final_layer'
# cascade applies.
#
# Normalization layers (input_layernorm, the final post-block norm, ...) sit
# between a residual writer and its consumer and are transparent to the
# hidden dim, so output_dependence routes through them ('final_norm' is the
# singleton sentinel for the post-block norm some families keep in
# extra_layers) and CoupledPruner.apply_chain forwards the mask through each
# normalization hop until it reaches a real matmul target.

VALID_METHODS = ["magnitude", "gradient", "random"]
VALID_REDUCE_OPS = ["add", "multiply"]
CALIBRATED_METHOD_DICT = {
    "gradient": ("weight_grad",),
}

# Keys used in the owning SchemeGroup's properties dict for shared-mask coordination.
_SHARED_MASK_KEY = "structured_pruning.shared_mask"


def _is_embedding(module) -> bool:
    """Whether the prunable (residual/embedding_dim) axis is weight's dim 1.

    Every other compressible module (LinearCompressed) keeps the prunable
    axis at weight's dim 0 ("out_features" rows). EmbeddingCompressed's
    ``weight`` is stored in nn.Embedding's native ``[num_embeddings,
    embedding_dim]`` layout (checkpoint-compatible), so its prunable axis
    (``embedding_dim``) is dim 1 instead -- everywhere below that reads/
    writes/removes rows branches on this.
    """
    return isinstance(module, EmbeddingCompressed)


class StructuredPruner(Compressor):
    def __init__(self, config):
        self.config = config
        self.ratio = self.config["ratio"]
        self.method = self.config.get("method", self.config.get("criterion", "magnitude"))
        self.granularity = self.config.get("granularity", "layer")
        self.repeated_pattern = self.config.get("repeated_pattern", False)
        self.share_mask = self.config.get("share_mask", False)
        self.reduce_op = self.config.get("reduce_op", None)
        self.calibration_store = None
        # Back-reference to the owning CompressionScheme, planted by the scheme
        # at construction time. Gives access to group siblings and the model's
        # generic pruning indexing (output_dependence) for the coupled cascade.
        self.scheme = None

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        if not self._to_compress():
            return ()
        return CALIBRATED_METHOD_DICT.get(self.method, ())

    def needs_grouping(self):
        # A shared mask must reduce scores across sibling layers, which is only
        # reachable through group membership.
        return bool(self.share_mask) and self._to_compress()

    def check_coupling(self, hard, siblings):
        # Only hard removal actually resizes tensors; soft pruning just zeroes
        # rows in place, so a mismatch between coupled layers can't manifest.
        if not hard or not siblings:
            return

        path = self.scheme.path if self.scheme is not None else "<unknown>"
        states = [(path, self._to_compress(), self.share_mask)]
        states += [
            (s.scheme.path if s.scheme is not None else "<unknown>", s._to_compress(), s.share_mask)
            for s in siblings
        ]

        active = [p for p, is_active, _ in states if is_active]
        inactive = [p for p, is_active, _ in states if not is_active]
        if active and inactive:
            raise ValueError(
                f"Coupled pruning: {active} are being hard-pruned but {inactive} "
                "(coupled via pruning.coupled_masks) are not. Prune all members of a "
                "coupled_masks group together -- call manager.auto_groups() and set "
                "share_mask=True on the resulting group -- or leave every member of "
                "the group unpruned."
            )

        if active and not all(shares_mask for _, _, shares_mask in states):
            raise ValueError(
                f"Coupled pruning: {active} are all being hard-pruned but do not share "
                "one mask (share_mask is not enabled on all of them), so their masks "
                "may select different neurons. Call manager.auto_groups() and set "
                "share_mask=True on the resulting group before apply(hard=True)."
            )

    # ------------------------------------------------------------------ scoring

    def get_scores(self, module):
        """Return the 1-D output-neuron importance scores for ``module``.

        Pure scoring: no mask logic. Used by ``apply`` and, for shared masks, by
        sibling compressors' first-in-group orchestration.
        """
        method = self.method
        score_fn = SCORE_FUNCTIONS.get(method)
        if score_fn is None:
            raise ValueError(f"Unsupported structured pruning method '{method}'.")
        weight = module.weight.t() if _is_embedding(module) else module.weight
        if method == "gradient":
            weight_grad = self.calibration_store["weight_grad"]
            weight_grad = weight_grad.t() if _is_embedding(module) else weight_grad
            return score_fn(weight, weight_grad)
        return score_fn(weight)

    # ------------------------------------------------------------------- masks

    def _pattern_keep_mask(self, module):
        """Length-``granularity`` keep-mask reduced across this layer's own groups."""
        scores = self.get_scores(module)
        pattern_scores = reduce_pattern_scores(scores, self.granularity, self.reduce_op)
        # Prune int(ratio * granularity) positions within the single pattern.
        return build_structured_mask(pattern_scores, self.ratio, "layer")

    def _own_mask(self, module):
        out_dim = module.weight.shape[1] if _is_embedding(module) else module.weight.shape[0]
        if self.repeated_pattern:
            pattern_mask = self._pattern_keep_mask(module)
            return tile_pattern_mask(pattern_mask, out_dim)
        scores = self.get_scores(module)
        return build_structured_mask(scores, self.ratio, self.granularity)

    def _group(self):
        """Return the single SchemeGroup this scheme belongs to (share_mask path)."""
        if self.scheme is None or not getattr(self.scheme, "groups", None):
            raise RuntimeError(
                "share_mask is enabled but the scheme is not part of any group. "
                "Create a group (e.g. manager.auto_groups()) before applying."
            )
        # For now a scheme belongs to at most one group.
        return next(iter(self.scheme.groups.values()))

    def _resolve_shared_mask(self, module):
        """First-in-group computes the reduced shared mask; others reuse it.

        The reduced result is stored in the group's properties dict so every
        member reuses it. Two kinds:

        * ``"full"`` — a full-length keep-mask shared verbatim; requires all
          members to have the same output dimension (e.g. gate/up).
        * ``"pattern"`` — a length-``granularity`` pattern (repeated_pattern),
          which each member tiles up to its own output dimension. This lets
          members with a *different* number of groups share a mask (GQA q/k).
        """
        group = self._group()

        cached = group.properties.get(_SHARED_MASK_KEY, None)
        if cached is None:
            cached = self._compute_group_shared(group)
            group.properties[_SHARED_MASK_KEY] = cached

        kind, mask = cached["kind"], cached["mask"]
        if kind == "pattern":
            return tile_pattern_mask(mask, module.weight.shape[0])
        return mask

    def _compute_group_shared(self, group):
        """Reduce every member's scores into the group's shared mask/pattern."""
        if self.repeated_pattern:
            # Each member contributes a length-granularity pattern; reduce across
            # members, then build one pattern mask that all members tile.
            pattern_list = []
            for sibling in group.schemes:
                compressor = sibling.compressors.get("structured_pruning", None)
                if compressor is None:
                    continue
                sibling_module = sibling.get_compression_module()
                sibling_scores = compressor.get_scores(sibling_module)
                pattern_list.append(
                    reduce_pattern_scores(sibling_scores, compressor.granularity, compressor.reduce_op)
                )
            reduced = reduce_scores(pattern_list, self.reduce_op)
            pattern_mask = build_structured_mask(reduced, self.ratio, "layer")
            return {"kind": "pattern", "mask": pattern_mask}

        # Full shared mask: members must share the same output dimension.
        score_list = []
        for sibling in group.schemes:
            compressor = sibling.compressors.get("structured_pruning", None)
            if compressor is None:
                continue
            sibling_module = sibling.get_compression_module()
            score_list.append(compressor.get_scores(sibling_module))
        reduced = reduce_scores(score_list, self.reduce_op)
        mask = build_structured_mask(reduced, self.ratio, self.granularity)
        return {"kind": "full", "mask": mask}

    def _resolve_mask(self, module):
        if self.share_mask:
            return self._resolve_shared_mask(module)
        return self._own_mask(module)

    # ------------------------------------------------------------------- apply

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return

        ratio = self.ratio
        method = self.method
        granularity = self.granularity

        validate_structured_pruning_ratio(ratio)
        validate_structured_pruning_method(method)
        validate_structured_pruning_granularity(granularity)
        validate_structured_pruning_repeated_pattern(self.repeated_pattern)
        validate_structured_pruning_share_mask(self.share_mask)
        validate_structured_pruning_reduce_op(self.reduce_op)

        if self.repeated_pattern and not isinstance(granularity, int):
            raise ValueError(
                "repeated_pattern requires an integer 'granularity' (the pattern size, "
                f"e.g. head_dim); got granularity={granularity!r}."
            )

        self.config["ratio"] = ratio
        self.config["method"] = method
        self.config["granularity"] = granularity

        if ratio <= 0:
            return

        # Determine the keep-mask. When finalizing an already-soft-applied module,
        # reuse the mask registered during the soft pass so hard removal is
        # consistent with what was zeroed.
        if soft_applied:
            mask = module._buffers.get("weight_mask", None)
            if mask is None:
                mask = self._resolve_mask(module)
        else:
            mask = self._resolve_mask(module)

        if not soft_applied:
            # Soft effect (STE): register the mask buffer and zero pruned rows.
            # Runs for both a soft apply and the first hard apply.
            module.register_buffer("weight_mask", mask)
            with torch.no_grad():
                dtype = module.weight.dtype
                if _is_embedding(module):
                    module.weight.mul_(mask.unsqueeze(0).to(dtype))
                else:
                    module.weight.mul_(mask.unsqueeze(1).to(dtype))
                    if module.bias is not None:
                        module.bias.mul_(mask.to(dtype))

        if hard:
            # Actual neuron removal + matrix resizing, then cascade the input
            # pruning to any dependent (next) layers.
            self._hard_remove(module, mask)
            self._cascade_coupled(mask, hard=True)

    def _hard_remove(self, module, mask):
        """Remove pruned output rows and resize weight/bias/out_features in place."""
        keep = mask.to(torch.bool)

        if _is_embedding(module):
            # Prunable axis is dim 1 (embedding_dim); vocab (dim 0) is untouched.
            with torch.no_grad():
                new_weight = module.weight.detach()[:, keep].clone()
            module.weight = nn.Parameter(new_weight, requires_grad=module.weight.requires_grad)
            module.embedding_dim = int(keep.sum().item())
            module.out_features = module.embedding_dim
        else:
            with torch.no_grad():
                new_weight = module.weight.detach()[keep, :].clone()
            module.weight = nn.Parameter(new_weight, requires_grad=module.weight.requires_grad)

            if module.bias is not None:
                with torch.no_grad():
                    new_bias = module.bias.detach()[keep].clone()
                module.bias = nn.Parameter(new_bias, requires_grad=module.bias.requires_grad)

            module.out_features = int(keep.sum().item())

        # The 1-D mask no longer matches the resized weight; drop it.
        module._buffers.pop("weight_mask", None)

    @staticmethod
    def _hop_block_id(path_list, source_name, source_block_id, target_name):
        """Resolve which block a coupled hop lives in, given its source hop.

        ``output_dependence`` entries are declared per-block, at the granularity
        of a single block's layer list (``path_list``). A dependency stays within
        the same block if the target comes *later* in that per-block execution
        order (e.g. self_attn.o_proj -> post_attention_layernorm, both consumed
        within this block via the residual stream). If the target comes at the
        same position or *earlier* (e.g. mlp.down_proj -> input_layernorm), the
        dependency wraps around: this block's output feeds the *next* block's
        input via the residual stream, so the coupled cascade must target
        ``block_id + 1``.

        Every output-writing layer touches every later block too (residual
        skip connections), but ``coupled_masks_all`` already forces one shared
        mask across all blocks' residual writers, so cascading strictly one
        block ahead is sufficient -- that next block's own cascade continues
        the chain from there. Generalized from a single scheme's own
        name/block_id so it can also resolve hops *within* a chain (e.g. a
        norm forwarding to its own output_dependence targets), not just the
        scheme's immediate targets.
        """
        if not path_list or source_block_id is None:
            return source_block_id
        try:
            source_pos = path_list.index(source_name)
            target_pos = path_list.index(target_name)
        except ValueError:
            return source_block_id
        return source_block_id + 1 if target_pos <= source_pos else source_block_id

    @staticmethod
    def _resolve_named_module(scheme, name, block_id, num_blocks):
        """Resolve a single ``output_dependence`` entry name to a module.

        Handles the ``'final_layer'``/``'final_norm'`` singleton sentinels
        (resolved off the scheme's pre-threaded paths, not block-indexed) and
        ordinary ``path_list`` leaf names (block-indexed via
        ``path_template``, dropped if out of range). Returns ``None`` if the
        entry can't be resolved (missing sentinel path, block out of range,
        or -- for the sentinels -- ``block_id`` isn't the last block).

        The last-block gating for the sentinels applies regardless of how
        deep in a chain they're reached: a source's own ``output_dependence``
        entry may name ``'final_layer'``/``'final_norm'`` directly (e.g.
        llama_c's ``mlp.down_proj``), or a normalization hop's *own*
        ``output_dependence`` entry may (e.g. bert_c's post-norm
        ``output.LayerNorm``, resolved through the recursion in
        ``_resolve_target_chains``) -- both must only fire once the walk has
        reached the last block, tracked via the ``block_id`` threaded through
        every hop by ``_hop_block_id``.
        """
        from ..utils.utils import get_submodule  # lazy: avoid import cycle

        if name in ("final_layer", "final_norm"):
            is_last_block = (
                num_blocks is not None and block_id is not None and block_id == num_blocks - 1
            )
            if not is_last_block:
                return None
            sentinel_path = scheme.final_layer_path if name == "final_layer" else scheme.final_norm_path
            if not sentinel_path:
                return None
            return get_submodule(scheme.model, sentinel_path)

        if num_blocks is not None and (block_id is None or block_id >= num_blocks):
            return None
        full_path = scheme.path_template.format(block_index=block_id, path=name)
        return get_submodule(scheme.model, full_path)

    def _resolve_target_chains(self, scheme, name, block_id):
        """Resolve one ``output_dependence`` entry to a list of hop chains.

        A chain is ``[hop_1, ..., terminal]``: normalization-layer hops
        (per indexing's ``pruning.norm_layers``, or the ``'final_norm'``
        sentinel) are transparent, so resolution walks through them via
        their *own* ``output_dependence`` entry -- which may fan out to
        several onward targets, producing multiple chains rooted at the same
        top-level name (e.g. ``input_layernorm`` -> q/k/v).
        """
        num_blocks = getattr(scheme, "num_blocks", None)
        module = self._resolve_named_module(scheme, name, block_id, num_blocks)
        if module is None:
            return []

        pruning_indexing = getattr(scheme, "pruning_indexing", None) or {}
        norm_layers = pruning_indexing.get("norm_layers", ())
        if name not in norm_layers and name != "final_norm":
            return [[module]]

        onward_names = pruning_indexing.get("output_dependence", {}).get(name, [])
        path_list = getattr(scheme, "path_list", None)
        chains = []
        for onward_name in onward_names:
            onward_block_id = self._hop_block_id(path_list, name, block_id, onward_name)
            for sub_chain in self._resolve_target_chains(scheme, onward_name, onward_block_id):
                chains.append([module] + sub_chain)
        return chains

    def _coupled_target_chains(self):
        """Resolve dependent (next) layer chains from generic pruning indexing.

        Model specifics come only from the scheme's ``pruning_indexing``
        (``output_dependence``); resolution/invocation stay in this compressor.

        One sentinel source gets special resolution: ``'preprocessing'`` --
        the singleton embedding scheme itself (``scheme.block_id is None``)
        always cascades into block 0. Every other name (including the
        ``'final_layer'``/``'final_norm'`` sentinels, wherever they appear)
        resolves generically through ``_resolve_target_chains``, which
        applies the last-block gating itself (see
        ``_resolve_named_module``).
        """
        scheme = self.scheme
        if scheme is None:
            return []
        pruning_indexing = getattr(scheme, "pruning_indexing", None) or {}
        dependence = pruning_indexing.get("output_dependence", {})
        next_layers = dependence.get(scheme.name, [])
        if not next_layers:
            return []

        num_blocks = getattr(scheme, "num_blocks", None)
        chains = []
        for next_layer in next_layers:
            target_block_id = 0 if scheme.block_id is None else self._hop_block_id(
                scheme.path_list, scheme.name, scheme.block_id, next_layer
            )
            if num_blocks is not None and target_block_id is not None and target_block_id >= num_blocks:
                # No next block to cascade into (ordinary path_list target).
                continue
            chains.extend(self._resolve_target_chains(scheme, next_layer, target_block_id))
        return chains

    def _cascade_coupled(self, mask, hard):
        for chain in self._coupled_target_chains():
            CoupledPruner().apply_chain(chain, mask, hard=hard)

    # ----------------------------------------------------------------- restore

    def restore(self, module):
        # Reset config only. The weight_mask buffer is intentionally left on the
        # module so the same mask can be re-applied after restoration.
        self.config["ratio"] = 0.0
        self.config["method"] = "magnitude"

        # Undo coupled input pruning on dependent layers.
        for chain in self._coupled_target_chains():
            CoupledPruner().revert_chain(chain)

        # Clear any shared mask left on the owning group so a later apply recomputes.
        if self.scheme is not None:
            for group in getattr(self.scheme, "groups", {}).values():
                group.properties.pop(_SHARED_MASK_KEY, None)

    def reapply_mask(self, module):
        """Re-zero pruned rows using the stored weight_mask buffer (STE fine-tuning)."""
        mask = module._buffers.get("weight_mask")
        if mask is None:
            return
        with torch.no_grad():
            dtype = module.weight.dtype
            if _is_embedding(module):
                module.weight.mul_(mask.unsqueeze(0).to(dtype))
            else:
                module.weight.mul_(mask.unsqueeze(1).to(dtype))
                if module.bias is not None:
                    module.bias.mul_(mask.to(dtype))

    def remove_mask(self, module):
        """Drop the weight_mask buffer from the module."""
        module._buffers.pop("weight_mask", None)

    def _to_compress(self):
        return self.ratio > 0.0

    def __repr__(self):
        return (
            f"StructuredPruner(ratio={self.ratio}, method='{self.method}', "
            f"granularity={self.granularity!r}, repeated_pattern={self.repeated_pattern}, "
            f"share_mask={self.share_mask}, reduce_op={self.reduce_op!r})"
        )


### CONFIGURATION VALIDATORS ###

def validate_structured_pruning_ratio(ratio: float) -> None:
    if ratio is not None and not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Pruning ratio must be between 0.0 and 1.0, but got {ratio}.")

def validate_structured_pruning_method(method: str) -> None:
    if method is not None and method not in VALID_METHODS:
        raise ValueError(f"Pruning method must be one of {VALID_METHODS}, but got '{method}'.")

def validate_structured_pruning_granularity(granularity) -> None:
    if granularity in ("layer", None):
        return
    if isinstance(granularity, bool) or not isinstance(granularity, int):
        raise ValueError(
            f"Pruning granularity must be 'layer' or a positive integer, but got {granularity!r}."
        )
    if granularity <= 0:
        raise ValueError(f"Pruning granularity must be a positive integer, but got {granularity}.")

def validate_structured_pruning_repeated_pattern(repeated_pattern) -> None:
    if not isinstance(repeated_pattern, bool):
        raise ValueError(f"repeated_pattern must be a boolean, but got {type(repeated_pattern)}.")

def validate_structured_pruning_share_mask(share_mask) -> None:
    if not isinstance(share_mask, bool):
        raise ValueError(f"share_mask must be a boolean, but got {type(share_mask)}.")

def validate_structured_pruning_reduce_op(reduce_op) -> None:
    if reduce_op in (None, ""):
        return
    if reduce_op not in VALID_REDUCE_OPS:
        raise ValueError(
            f"reduce_op must be one of {VALID_REDUCE_OPS} (or None), but got '{reduce_op}'."
        )

__all__ = [
    "StructuredPruner",
    "validate_structured_pruning_ratio",
    "validate_structured_pruning_method",
    "validate_structured_pruning_granularity",
    "validate_structured_pruning_repeated_pattern",
    "validate_structured_pruning_share_mask",
    "validate_structured_pruning_reduce_op",
]
