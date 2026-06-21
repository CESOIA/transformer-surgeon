"""Base summary contracts and runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple

import torch


class CalibrationSummary(ABC):
    """Abstract contract for a statistic computed over calibration batches.

    ``CalibrationSummary`` instances are **singletons** stored in
    ``SUMMARY_REGISTRY``. A single instance is shared across all schemes; per-
    scheme accumulation state lives in a companion
    :class:`SummaryRuntime` object managed by the backbone.

    When the backbone runs a calibration pass it keeps one ``SummaryRuntime``
    per *(scheme, summary)* pair. As each calibration batch is processed, raw
    tensors emitted by :class:`~transformersurgeon.calibration.raw_data.base.RawDataCollector`
    hooks are delivered to ``SummaryRuntime.on_raw``. Once a complete payload
    for all ``required_raw_data`` streams has arrived, the runtime calls
    :meth:`update_runtime`, which writes the result into the per-scheme
    ``calibration_store`` dict. After all batches are done,
    :meth:`finalize_store` is called once.

    The corresponding ``Compressor`` later reads the result from
    ``calibration_store[self.name]`` inside its ``apply`` method.

    To add a new summary type:

    1. Subclass ``CalibrationSummary``, set ``name`` and
       ``required_raw_data``, and implement :meth:`update_from_raw`.
    2. Override :meth:`update_runtime` if you need running statistics across
       batches (e.g., token-weighted averages — see ``CovarianceSummary``).
    3. Add a singleton instance to ``SUMMARY_REGISTRY`` in
       ``calibration/summaries/registry.py``.
    4. Reference ``self.name`` in your ``Compressor.needs_calibration()``
       return value.
    """

    name: str
    """Registry key for this summary.  Used as the lookup key in
    ``SUMMARY_REGISTRY`` and as the storage key in ``calibration_store``.
    Must match the string returned by :meth:`~transformersurgeon.compression.abstract.Compressor.needs_calibration`."""

    required_raw_data: Tuple[str, ...]
    """Names of the raw data streams that must all arrive before one summary
    update can fire (matched in FIFO order by :class:`SummaryRuntime`).  Each
    name must be a key in ``RAW_DATA_REGISTRY``."""

    requires_shifted_model: bool = False
    """If ``True``, the backbone spins up a shifted copy of the model and
    registers collectors whose ``uses_shifted_model`` flag is ``True`` on that
    copy. Required for cross-activation summaries (e.g., AA-SVD)."""

    def initialize_store(self, calibration_store: dict) -> None:
        """Reset this summary's namespace in the calibration store.

        Called once before the calibration pass begins. The default
        implementation removes only the key owned by this summary
        (``calibration_store.pop(self.name, None)``), leaving other summaries'
        data untouched. Override if your summary uses multiple keys or requires
        a non-destructive init.

        Args:
            calibration_store: The per-scheme dict that accumulates results.
        """
        # Reset only this summary namespace, keep other summaries untouched.
        calibration_store.pop(self.name, None)

    @abstractmethod
    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        """Update the summary from one complete raw-data payload.

        Called by the default :meth:`update_runtime` implementation once per
        matched payload. ``raw_data`` keys are exactly the strings in
        ``self.required_raw_data``. Write your result into
        ``calibration_store[self.name]``.

        If you need to accumulate across batches (e.g., a running average),
        override :meth:`update_runtime` instead — it provides access to
        ``runtime.state`` for bookkeeping. The default ``update_runtime``
        calls this method directly, which means each call **overwrites** the
        previous value (single-batch behavior).

        Args:
            calibration_store: The per-scheme dict; write results here.
            raw_data: Mapping of stream name → tensor for one matched payload.
                Keys are exactly ``self.required_raw_data``.
        """

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        """Stateful update hook called once per matched payload.

        Override this (instead of :meth:`update_from_raw`) when you need
        running statistics across multiple batches. The ``runtime`` object
        provides:

        * ``runtime.state`` — a free-form ``dict`` you can use for
          bookkeeping (e.g., token counts for weighted averages).
        * ``runtime.num_updates`` — how many payloads have been processed so
          far for this scheme.

        See ``CovarianceSummary.update_runtime`` for an example that maintains
        a token-weighted running average using ``runtime.state["token_count"]``.

        The default implementation ignores ``runtime`` and delegates directly
        to :meth:`update_from_raw`, which overwrites the stored value each
        call (suitable only for single-batch calibration).

        Args:
            runtime: :class:`SummaryRuntime` instance holding per-scheme
                accumulation state.
            calibration_store: The per-scheme dict; write results here.
            raw_data: Mapping of stream name → tensor for one matched payload.
        """
        # Default implementation ignores runtime state and performs a direct overwrite.
        self.update_from_raw(calibration_store, raw_data)

    def finalize_store(self, calibration_store: dict) -> None:
        """Post-process the calibration store after all batches are done.

        Called once after the calibration loop completes. Use for
        normalization, regularization, or any other finalization step. The
        default implementation is a no-op.

        Args:
            calibration_store: The per-scheme dict containing accumulated
                summary values.
        """


@dataclass
class SummaryRuntime:
    """Per-scheme accumulation state for one ``CalibrationSummary`` instance.

    This is an **internal dataclass** managed entirely by the calibration
    backbone. You do not subclass or instantiate it directly.

    Because ``CalibrationSummary`` instances are singletons (shared across all
    schemes), per-scheme state is tracked here instead. One ``SummaryRuntime``
    is created per *(scheme, summary)* pair at the start of every calibration
    pass.

    Attributes:
        summary: The singleton ``CalibrationSummary`` this runtime wraps.
        pending_raw: FIFO buffers keyed by raw stream name. Incoming tensors
            are appended; a payload is dequeued from all streams together once
            every stream has at least one item.
        num_updates: Running count of payloads delivered to the summary so far.
        state: Free-form dict available to ``summary.update_runtime`` for
            bookkeeping (e.g., ``state["token_count"]`` in covariance
            summaries).
    """

    summary: CalibrationSummary
    pending_raw: Dict[str, list[torch.Tensor]] = field(default_factory=dict)
    num_updates: int = 0
    state: Dict[str, int] = field(default_factory=dict)

    def reset_batch(self) -> None:
        # Clear queues between batches; leftovers indicate stream pairing mismatch.
        leftovers = {
            raw_name: len(queue)
            for raw_name, queue in self.pending_raw.items()
            if len(queue) > 0
        }
        if len(leftovers) > 0:
            raise RuntimeError(
                "Calibration raw stream mismatch: unpaired raw payloads remained at batch boundary "
                f"for summary {self.summary.name}: {leftovers}."
            )
        self.pending_raw.clear()

    def on_raw(self, calibration_store: dict, raw_name: str, raw_value: torch.Tensor) -> None:
        # Ignore raw streams unrelated to this summary.
        if raw_name not in self.summary.required_raw_data:
            return

        self.pending_raw.setdefault(raw_name, []).append(raw_value)

        # Update while complete FIFO payloads are available across required streams.
        while all(len(self.pending_raw.get(name, [])) > 0 for name in self.summary.required_raw_data):
            payload = {
                name: self.pending_raw[name].pop(0)
                for name in self.summary.required_raw_data
            }
            self.summary.update_runtime(self, calibration_store, payload)
            self.num_updates += 1


def unique_summaries(summary_names: Iterable[str]) -> Tuple[str, ...]:
    # Order-preserving de-duplication for deterministic calibration behavior.
    unique = []
    for summary_name in summary_names:
        if summary_name not in unique:
            unique.append(summary_name)
    return tuple(unique)
