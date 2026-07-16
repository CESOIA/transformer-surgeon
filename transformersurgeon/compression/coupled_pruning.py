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
    ELEMENTWISE_BACKUP_ATTR = "_coupled_elementwise_backup"

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

    # ------------------------------------------------------------- chains

    @staticmethod
    def _is_elementwise(module) -> bool:
        """Whether ``module`` is a normalization layer, not a matmul target.

        Purely structural: normalization layers (RMSNorm/LayerNorm) have a
        1-D affine ``weight`` (length = hidden size); every matmul target
        (``LinearCompressed``/``EmbeddingCompressed``) always has a 2-D
        ``weight``. This keeps ``CoupledPruner`` model-agnostic -- it reacts
        to a shape fact, not to indexing metadata or module identity.
        """
        weight = getattr(module, "weight", None)
        return weight is not None and weight.dim() == 1

    def _apply_elementwise(self, module, mask):
        """Prune a normalization layer's own 1-D weight/bias in place."""
        keep = mask.to(torch.bool)

        existing = getattr(module, self.ELEMENTWISE_BACKUP_ATTR, None)
        if existing is not None:
            if existing["mask"].shape == keep.shape and torch.equal(existing["mask"], keep):
                return
            raise ValueError(
                f"Conflicting coupled pruning masks target normalization layer {module}. "
                "Layers that share a downstream normalization target must share one mask "
                "(share_mask)."
            )

        if keep.numel() != module.weight.numel():
            raise ValueError(
                f"Coupled pruning mask length {keep.numel()} does not match the "
                f"{module.weight.numel()} channels of normalization layer {module}."
            )

        backup = {
            "weight": module.weight.detach().clone(),
            "mask": keep.detach().clone(),
        }
        has_bias = getattr(module, "bias", None) is not None
        if has_bias:
            backup["bias"] = module.bias.detach().clone()
        has_normalized_shape = hasattr(module, "normalized_shape")
        if has_normalized_shape:
            backup["normalized_shape"] = module.normalized_shape
        setattr(module, self.ELEMENTWISE_BACKUP_ATTR, backup)

        with torch.no_grad():
            module.weight = nn.Parameter(
                module.weight.detach()[keep].clone(), requires_grad=module.weight.requires_grad
            )
            if has_bias:
                module.bias = nn.Parameter(
                    module.bias.detach()[keep].clone(), requires_grad=module.bias.requires_grad
                )
        if has_normalized_shape:
            module.normalized_shape = (int(keep.sum().item()),)

    def _revert_elementwise(self, module):
        backup = getattr(module, self.ELEMENTWISE_BACKUP_ATTR, None)
        if backup is None:
            return

        with torch.no_grad():
            module.weight = nn.Parameter(
                backup["weight"].clone(), requires_grad=module.weight.requires_grad
            )
            if "bias" in backup:
                module.bias = nn.Parameter(
                    backup["bias"].clone(), requires_grad=module.bias.requires_grad
                )
        if "normalized_shape" in backup:
            module.normalized_shape = backup["normalized_shape"]
        delattr(module, self.ELEMENTWISE_BACKUP_ATTR)

    def apply_chain(self, chain, mask, hard=True):
        """Prune a resolved hop chain, forwarding through normalization layers.

        ``chain`` is ``[hop_1, hop_2, ..., terminal]``: zero or more
        normalization-layer hops (transparent to the embedding/residual
        dimension -- pruned elementwise and passed through unchanged)
        followed by exactly one real matmul target (pruned as ``apply()``
        already does). Recurses through the chain: each normalization hop
        checks whether pruning needs to be forwarded further and, if so,
        calls coupled pruning on the subsequent layer.

        Args:
            chain: Ordered list of modules from ``StructuredPruner``'s
                output_dependence resolution; empty is a no-op.
            mask: 1-D boolean keep-mask, constant across the whole chain
                (normalization layers don't change the embedding size).
            hard: Coupled pruning only acts in hard mode. ``False`` is a no-op.
        """
        if not hard or not chain:
            return

        head, *rest = chain
        if getattr(head, "skip", False):
            return

        if self._is_elementwise(head):
            self._apply_elementwise(head, mask)
            if rest:
                self.apply_chain(rest, mask, hard=hard)
            return

        self.apply(head, mask, hard=hard)

    def revert_chain(self, chain):
        """Revert every hop of a chain previously pruned by :meth:`apply_chain`."""
        for module in chain:
            if self._is_elementwise(module):
                self._revert_elementwise(module)
            else:
                self.revert(module)


__all__ = ["CoupledPruner"]
