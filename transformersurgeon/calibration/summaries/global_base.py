"""Global calibration summaries: statistics computed across layers once the calibration pass is done."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence, Tuple


class GlobalCalibrationSummary(ABC):
    """A statistic that needs the per-scheme summaries of *several* schemes at once.

    Per-scheme summaries (``CalibrationSummary``) are accumulated over the calibration batches, one dict per
    scheme. A global summary runs after all of them are finalized: it reads the per-scheme dicts of the schemes
    that were calibrated for it (those whose store holds every name in :attr:`requires`) and writes one shared
    result into the manager-level ``global_store`` under :attr:`name`. Typical use: standardise per-layer scores
    and choose a threshold over all layers (FLAP).

    To add one: subclass, set ``name`` and ``requires``, implement :meth:`compute`, and register an instance in
    ``GLOBAL_SUMMARY_REGISTRY`` (``summaries/registry.py``). A compressor asks for it by returning
    ``{name: options}`` from ``needs_global_calibration()``.
    """

    name: str
    requires: Tuple[str, ...]
    """Per-scheme summaries that must be present in a scheme's store for the scheme to take part."""

    @abstractmethod
    def compute(self, entries: Sequence[tuple], options: Mapping, global_store: dict) -> None:
        """Write the global result into ``global_store[self.name]``.

        Args:
            entries: ``[(scheme, scheme_calibration_store), ...]`` for every participating scheme.
            options: the options the requesting compressors asked for (identical across requesters).
            global_store: the shared dict to write into.
        """
