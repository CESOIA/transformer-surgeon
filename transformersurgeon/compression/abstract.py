from abc import ABC, abstractmethod


class Compressor(ABC):
    """Abstract base class for all compression algorithms.

    A ``Compressor`` encapsulates one compression method (e.g., low-rank
    decomposition, pruning, quantization) and its configuration. It is
    always paired with a single ``LinearCompressed`` module via a
    ``CompressionScheme``.

    Lifecycle (called in this order by the manager):

    1. ``set_calibration_store(calibration_data)`` — called before anything
       else; stores the per-scheme calibration dict.
    2. ``needs_calibration()`` — the manager inspects the return value to
       decide whether to run a calibration pass before ``apply``.
    3. *(manager runs calibration if needed, populating the store)*
    4. ``apply(module)`` — compress the module.
    5. ``restore(module)`` — optionally reverse the compression.

    Concrete implementations live under ``compression/*_methods/`` and are
    registered in ``compression/registry.py``.
    """

    @abstractmethod
    def set_calibration_store(self, calibration_data):
        """Store a reference to the scheme-level calibration summary dict.

        The manager calls this method automatically before ``needs_calibration``
        and ``apply``. The compressor must store the reference (not a copy) so
        that it always sees the most-recent values written by the calibration
        backbone.

        Args:
            calibration_data: The per-scheme ``dict`` shared between the
                manager and this compressor. Summary values are keyed by
                ``CalibrationSummary.name`` (e.g. ``"covariance"``). The
                compressor reads from this dict inside ``apply``.
        """
        pass

    @abstractmethod
    def needs_calibration(self):
        """Return the calibration summaries required before ``apply`` can run.

        The manager inspects the return value to determine whether to schedule
        a calibration pass. An empty tuple means the compressor can apply
        without any data-dependent pre-computation (e.g., plain SVD). A
        non-empty tuple lists the ``CalibrationSummary.name`` keys that must
        be present in the calibration store before ``apply`` is called.

        Returns:
            tuple[str, ...]: Names of required summaries, or ``()`` if none.

        Examples — ``LRDer`` return values by method:

        * ``"svd"``        → ``()``
        * ``"svd-llm-v2"`` → ``("covariance",)``
        * ``"aa-svd"``     → ``("cross_covariance", "shifted_covariance")``
        """
        pass

    @abstractmethod
    def apply(self, module, hard=False, soft_applied=False):
        """Apply compression to the given ``LinearCompressed`` module.

        The calibration store must already be populated (if this compressor
        requires calibration) before this method is called.

        Three-way behavior controlled by the flag combination:

        * ``hard=False`` (default) — reversible (soft) compression. The
          module's topology may change (e.g., ``init_lrd`` adds ``weight_2``),
          but ``restore`` can fully undo it.
        * ``hard=True, soft_applied=False`` — apply compression directly in
          its final, irreversible form (topology change is permanent).
        * ``hard=True, soft_applied=True`` — finalize an already-soft-applied
          module. For methods where soft and hard are identical (e.g., LRD),
          this is a no-op.

        Args:
            module: A ``LinearCompressed`` instance to compress. Never a plain
                ``nn.Linear``.
            hard (bool): If ``True``, the compression is final and structural
                changes cannot be undone by ``restore``. Defaults to ``False``.
            soft_applied (bool): If ``True``, the module has already been
                soft-compressed; used to detect the finalize-in-place case.
                Defaults to ``False``.
        """
        pass

    @abstractmethod
    def restore(self, module):
        """Reverse the compression applied by ``apply``.

        The implementation must return ``module`` to its pre-``apply`` weight
        shape and remove any auxiliary parameters (e.g., ``weight_2`` for LRD).
        For LRD specifically: reconstruct ``weight = weight @ weight_2``, then
        call ``module.cancel_lrd()``. If ``_to_compress()`` returns ``False``
        the module was never modified, so restore should be a no-op.

        Args:
            module: The ``LinearCompressed`` instance to restore.
        """
        pass

    @abstractmethod
    def _to_compress(self):
        """Return whether compression should actually be applied.

        Acts as a guard checked by both ``apply`` and ``restore``. Implementations
        should return ``False`` when the primary configuration parameter
        represents an identity operation, making the compressor a no-op:

        * LRD: ``rank == "full"``
        * Pruning: ``ratio == 0.0``
        * Quantization: ``bits == 32`` (or equivalent full-precision sentinel)

        Returns:
            bool: ``True`` if compression should run; ``False`` to skip.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a human-readable string describing the compressor's config.

        The output appears in ``CompressionScheme.__repr__()`` and therefore
        in the printout produced by ``manager.print_filtered()``. Include all
        user-configurable fields so the printout is self-documenting.

        Example pattern::

            def __repr__(self):
                return f"MyCompressor(param1={self.param1}, param2={self.param2})"
        """
        pass


__all__ = ["Compressor"]
