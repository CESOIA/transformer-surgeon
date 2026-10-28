"""Block pruning: remove whole blocks of output rows (e.g. attention heads).

Separate from structured pruning, which removes individual rows/channels (and,
on attention, shrinks the per-head dimension). Block pruning removes complete
row blocks -- for ``self_attn.q_proj`` one block = one query head of
``head_dim`` rows -- together with the matching input-column blocks of the
layers that consume them (``o_proj``), and updates the owning module's head
bookkeeping so the HF model keeps running.

Model truth comes from the family's indexing, ``pruning.block_pruning``::

    'block_pruning': {
        'self_attn.q_proj': {
            'owner': 'self_attn',                            # module holding the head bookkeeping
            'block_size_attr': 'head_dim',                   # owner attr: rows per block
            'blocks_per_unit_attr': 'num_key_value_groups',  # owner attr: blocks sharing one unit (GQA group);
                                                             #   updated to the kept count on hard apply
            'input_dependence': ['self_attn.o_proj'],        # input columns follow the blocks 1:1
        },
    }

Blocks are removed **uniformly per unit** (the same number of query heads in
every KV group), so GQA stays regular: HF's ``repeat_kv`` and the converted
decoder's grouped attention need an equal number of query heads per KV head.
Units are consecutive (query heads of KV group g are heads
``g*per_unit .. (g+1)*per_unit - 1``).

Soft apply zeroes the removed heads' input columns of the dependent layers
(zeroing the query rows would not remove a head: softmax of zero scores is a
uniform average, not zero). Hard apply removes the query rows and the
dependent columns and records ``block_prune_mask`` (kept blocks) on the pruned
layer; ``convert_for_export`` reads the kept head count from the pruned shapes
(``blocks/mha.py::MHABase.finalize_head_pruning``).
"""
from __future__ import annotations

import torch

from .validators import in_range, one_of
from .abstract import Compressor
from .coupled_pruning import CoupledPruner

VALID_METHODS = ["magnitude", "random"]
_SOFT_BACKUP_ATTR = "_block_pruning_soft_backup"


def _block_scores_magnitude(q_weight, dep_weights, block, n_blocks):
    """Head importance = ||q block||_F * prod ||dependent column block||_F."""
    s = q_weight.detach().float().reshape(n_blocks, -1).norm(dim=1)
    for w in dep_weights:
        s = s * w.detach().float().t().reshape(n_blocks, -1).norm(dim=1)
    return s


class BlockPruner(Compressor):
    def __init__(self, config):
        self.config = config
        self.ratio = self.config.get("ratio", 0.0)
        self.method = self.config.get("method", "magnitude")
        self.blocks = self.config.get("blocks", None)
        self.calibration_store = None
        self.scheme = None

    # ------------------------------------------------------------ Compressor API

    def set_calibration_store(self, calibration_data):
        self.calibration_store = calibration_data

    def needs_calibration(self):
        return ()

    def _to_compress(self):
        return bool(self.blocks) or (self.ratio is not None and self.ratio > 0.0)

    def __repr__(self):
        return f"BlockPruner(ratio={self.ratio}, method='{self.method}', blocks={self.blocks})"

    # ------------------------------------------------------------------ geometry

    def _spec(self):
        scheme = self.scheme
        if scheme is None:
            raise RuntimeError("BlockPruner needs its CompressionScheme (set by the manager).")
        spec = (scheme.pruning_indexing.get("block_pruning", {}) or {}).get(scheme.name)
        if spec is None:
            raise ValueError(
                f"Block pruning is not defined for '{scheme.name}' in this model's indexing "
                "(pruning.block_pruning). Only layers listed there can be block-pruned."
            )
        return spec

    def _resolve(self, name):
        from ..utils import get_submodule
        scheme = self.scheme
        return get_submodule(scheme.model, scheme.path_template.format(block_index=scheme.block_id, path=name))

    def _geometry(self, module):
        spec = self._spec()
        owner = self._resolve(spec["owner"])
        block = int(getattr(owner, spec["block_size_attr"]))
        per_unit = int(getattr(owner, spec["blocks_per_unit_attr"]))
        if module.out_features % block:
            raise ValueError(f"{self.scheme.path}: out_features {module.out_features} is not a multiple of "
                             f"the block size {block}.")
        n_blocks = module.out_features // block
        if n_blocks % per_unit:
            raise ValueError(f"{self.scheme.path}: {n_blocks} blocks do not split into units of {per_unit}.")
        deps = [self._resolve(n) for n in spec.get("input_dependence", ())]
        return spec, owner, block, per_unit, n_blocks, deps

    def removed_blocks(self, module):
        """Sorted global indices of the blocks this compressor removes."""
        spec, owner, block, per_unit, n_blocks, deps = self._geometry(module)
        n_units = n_blocks // per_unit
        if self.blocks:
            removed = sorted({int(b) for b in self.blocks})
            if removed[0] < 0 or removed[-1] >= n_blocks:
                raise ValueError(f"{self.scheme.path}: block indices must be in [0, {n_blocks}); got {removed}.")
            per = [sum(1 for b in removed if b // per_unit == u) for u in range(n_units)]
            if len(set(per)) != 1:
                raise ValueError(
                    f"{self.scheme.path}: block pruning must remove the same number of blocks from every unit "
                    f"(units of {per_unit}, e.g. query heads per KV group); got per-unit counts {per}."
                )
        else:
            k = int(self.ratio * per_unit)
            if k == 0:
                return []
            if self.method == "magnitude":
                scores = _block_scores_magnitude(module.weight, [d.weight for d in deps], block, n_blocks)
            elif self.method == "random":
                scores = torch.rand(n_blocks)
            else:
                raise ValueError(f"Block pruning method must be one of {VALID_METHODS}; got '{self.method}'.")
            removed = []
            for u in range(n_units):
                local = scores[u * per_unit:(u + 1) * per_unit]
                removed += [u * per_unit + int(i) for i in torch.argsort(local)[:k]]
            removed = sorted(removed)
        if len(removed) // n_units >= per_unit:
            raise ValueError(f"{self.scheme.path}: removing every block of a unit is not supported "
                             "(it would remove the unit itself, e.g. a whole KV group).")
        return removed

    # --------------------------------------------------------------------- apply

    def apply(self, module, hard=False, soft_applied=False):
        if not self._to_compress():
            return
        validate_block_pruning_ratio(self.ratio)
        validate_block_pruning_method(self.method)
        validate_block_pruning_blocks(self.blocks)
        if getattr(self.scheme, "position_linked", False) and "rope_prune_mask" in module._buffers:
            raise NotImplementedError("Block pruning on top of structured (head_dim) pruning of the same "
                                      "layer is not supported.")
        spec, owner, block, per_unit, n_blocks, deps = self._geometry(module)
        removed = self.removed_blocks(module)
        self.config.update(ratio=self.ratio, method=self.method, blocks=self.blocks)
        if not removed:
            return
        keep_blocks = torch.ones(n_blocks, dtype=torch.bool)
        keep_blocks[torch.tensor(removed)] = False
        keep_rows = keep_blocks.repeat_interleave(block)

        if not hard:
            # Soft: silence the removed heads by zeroing their dependent input columns.
            for d in deps:
                target, _ = CoupledPruner()._input_target(d)
                if getattr(d, _SOFT_BACKUP_ATTR, None) is None:
                    setattr(d, _SOFT_BACKUP_ATTR, target.weight.detach().clone())
                with torch.no_grad():
                    target.weight[:, ~keep_rows] = 0
            module.register_buffer("block_prune_mask", keep_blocks.clone())
            return

        # Hard: restore any soft zeroing first, then remove rows/columns for real.
        self._undo_soft(deps)
        with torch.no_grad():
            module.weight = torch.nn.Parameter(module.weight.detach()[keep_rows].clone(),
                                               requires_grad=module.weight.requires_grad)
            if module.bias is not None:
                module.bias = torch.nn.Parameter(module.bias.detach()[keep_rows].clone(),
                                                 requires_grad=module.bias.requires_grad)
        module.out_features = int(keep_rows.sum())
        for d in deps:
            CoupledPruner().apply(d, keep_rows, hard=True)
        kept_per_unit = per_unit - len(removed) // (n_blocks // per_unit)
        setattr(owner, spec["blocks_per_unit_attr"], kept_per_unit)
        module.register_buffer("block_prune_mask", keep_blocks.clone())

    def _undo_soft(self, deps):
        for d in deps:
            backup = getattr(d, _SOFT_BACKUP_ATTR, None)
            if backup is not None:
                target, _ = CoupledPruner()._input_target(d)
                with torch.no_grad():
                    target.weight.copy_(backup)
                delattr(d, _SOFT_BACKUP_ATTR)

    def restore(self, module):
        """Undo a soft apply (hard block pruning is final)."""
        if self.scheme is not None:
            try:
                deps = self._geometry(module)[-1]
            except ValueError:
                deps = []
            self._undo_soft(deps)
        module._buffers.pop("block_prune_mask", None)
        self.config["ratio"] = 0.0
        self.config["blocks"] = None


### CONFIGURATION VALIDATORS ###

validate_block_pruning_ratio = in_range("Block pruning ratio (fraction of blocks per unit)", 0.0, 1.0, include_high=False)
validate_block_pruning_method = one_of("Block pruning method", VALID_METHODS, allow_none=True)

def validate_block_pruning_blocks(blocks) -> None:
    if blocks is None:
        return
    if isinstance(blocks, (str, bytes)) or not all(isinstance(b, int) and not isinstance(b, bool) for b in blocks):
        raise ValueError(f"'blocks' must be None or a list of block (e.g. head) indices, got {blocks!r}.")


__all__ = [
    "BlockPruner",
    "validate_block_pruning_ratio",
    "validate_block_pruning_method",
    "validate_block_pruning_blocks",
]
