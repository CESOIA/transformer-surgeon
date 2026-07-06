import torch
import torch.nn as nn


class CoupledPruner:
    """Prune the INPUT columns of a layer to match a source layer's pruned outputs.

    When structured pruning removes output neurons (rows) of a source layer, any
    layer that consumes those outputs must have the matching input columns removed
    so the shapes stay consistent. This class performs that downstream input
    pruning.

    Hard-only by design: in soft mode the source layer's zeroed output rows already
    produce zero activations, so the downstream layer needs no change. ``apply`` is
    invoked in cascade by :class:`StructuredPruner` during a hard apply, and
    ``revert`` is called by ``StructuredPruner.restore``.

    Model-agnostic: it never inspects layer semantics; it only resizes the input
    dimension of whichever module it is handed, honouring low-rank decomposition
    (resizes ``linear_V`` when LRD is active).
    """

    BACKUP_ATTR = "_coupled_input_backup"

    def _input_target(self, module):
        """Return the sub-module whose weight columns represent the input dim.

        Under LRD the input flows through ``linear_V`` (in -> rank), so the input
        columns live there; otherwise they live on ``module.weight`` directly.
        """
        lrd_active = (
            getattr(module, "linear_V", None) is not None
            and isinstance(getattr(module, "rank", "full"), int)
        )
        return (module.linear_V if lrd_active else module), lrd_active

    def apply(self, module, mask, hard=True):
        """Remove input columns of ``module`` selected by the source keep-``mask``.

        Args:
            module: The downstream ``LinearCompressed`` whose inputs to prune.
            mask: 1-D boolean keep-mask over the source layer's output rows;
                its length must equal ``module``'s input-feature count.
            hard: Coupled pruning only acts in hard mode. ``False`` is a no-op.
        """
        if not hard:
            return
        if getattr(module, "skip", False):
            return

        keep = mask.to(torch.bool)

        # Idempotency: a target shared by several co-masked source layers (e.g.
        # gate_proj and up_proj both feeding down_proj) must be pruned only once.
        # If already coupled-pruned with the identical mask, skip; if a *different*
        # mask targets the same input, that is an unsupported conflict.
        existing = getattr(module, self.BACKUP_ATTR, None)
        if existing is not None:
            if existing["mask"].shape == keep.shape and torch.equal(existing["mask"], keep):
                return
            raise ValueError(
                f"Conflicting coupled pruning masks target the input of {module}. "
                "Layers that share a downstream target must share one mask (share_mask)."
            )

        target, lrd_active = self._input_target(module)
        in_features = target.in_features

        if keep.numel() != in_features:
            raise ValueError(
                f"Coupled pruning mask length {keep.numel()} does not match the "
                f"input features {in_features} of {module}."
            )

        # Backup so revert can restore the original input columns.
        setattr(module, self.BACKUP_ATTR, {
            "weight": target.weight.detach().clone(),
            "in_features": in_features,
            "lrd_active": lrd_active,
            "mask": keep.detach().clone(),
        })

        with torch.no_grad():
            new_weight = target.weight.detach()[:, keep].clone()
        target.weight = nn.Parameter(new_weight, requires_grad=target.weight.requires_grad)

        new_in = int(keep.sum().item())
        target.in_features = new_in
        module.in_features = new_in

    def revert(self, module):
        """Restore the original input columns backed up by :meth:`apply`.

        No-op if the module was never coupled-pruned.
        """
        backup = getattr(module, self.BACKUP_ATTR, None)
        if backup is None:
            return

        target = module.linear_V if backup["lrd_active"] else module
        with torch.no_grad():
            restored = backup["weight"].clone()
        target.weight = nn.Parameter(restored, requires_grad=target.weight.requires_grad)
        target.in_features = backup["in_features"]
        module.in_features = backup["in_features"]
        delattr(module, self.BACKUP_ATTR)


__all__ = ["CoupledPruner"]
