"""
Base raw data contracts and shared calibration batch utilities.
"""

from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

import torch


class RawDataCollector(ABC):
    """Stateless factory that produces PyTorch hooks for capturing raw tensor streams.

    A ``RawDataCollector`` is instantiated once per required raw stream (e.g.,
    one instance for ``"activation"``, another for ``"weight_grad"``). Its sole
    responsibility is to build a forward hook (or provide a backward callback)
    that captures a tensor from a ``LinearCompressed`` module and emits it
    upstream via the ``emit_raw`` callback. Accumulation and statistics are
    handled by :class:`~transformersurgeon.calibration.summaries.base.CalibrationSummary`,
    not by the collector.

    To add a new raw data stream:

    1. Subclass ``RawDataCollector`` and set the ``name`` class attribute.
    2. Implement ``build_forward_hook`` (and/or ``collect_after_backward``).
    3. Register an instance factory in ``calibration/raw_data/registry.py``
       under ``RAW_DATA_REGISTRY[name]``.

    The backbone then wires the hook to every target ``LinearCompressed`` module
    and routes emitted tensors to all ``SummaryRuntime`` objects that declared
    this stream in their ``required_raw_data``.
    """

    name: str
    """Registry key for this collector.  Must match the string used in
    :attr:`~transformersurgeon.calibration.summaries.base.CalibrationSummary.required_raw_data`
    and in ``RAW_DATA_REGISTRY``."""

    requires_backward: bool = False
    """If ``True``, the backbone will call :meth:`collect_after_backward` on
    each target module after ``loss.backward()`` for every calibration batch.
    Set this for gradient-based collectors (e.g., ``WeightGradCollector``)."""

    uses_shifted_model: bool = False
    """If ``True``, the backbone registers this collector's hook on a *shifted*
    copy of the model rather than the base model. Required for collectors that
    capture activations under a small positional offset (e.g.,
    ``ShiftedActivationCollector``). The backbone only spins up the shifted
    model when at least one required collector has this flag set."""

    def build_forward_hook(
        self,
        *,
        emit_raw: Callable[[str, torch.Tensor], None],
    ):
        """Return a PyTorch forward hook that emits tensors into the pipeline.

        The backbone calls this factory once per target module and registers the
        returned callable with ``module.register_forward_hook(...)``. Return
        ``None`` if this collector does not use a forward hook (e.g., it only
        operates post-backward via :meth:`collect_after_backward`).

        The hook signature must be::

            def hook(module, inputs, output) -> None: ...

        Inside the hook, call ``emit_raw`` exactly once per forward pass::

            emit_raw(self.name, tensor)

        ``emit_raw`` fans the tensor out to all
        :class:`~transformersurgeon.calibration.summaries.base.SummaryRuntime`
        objects that declared ``self.name`` in their ``required_raw_data``.

        Args:
            emit_raw: Callback provided by the backbone with signature
                ``(stream_name: str, tensor: torch.Tensor) -> None``.
                ``stream_name`` must equal ``self.name``.

        Returns:
            A callable hook ``(module, inputs, output) -> None``, or ``None``
            if this collector does not use a forward hook.
        """
        # Return None when collector does not use forward-hook collection.
        return None

    def collect_after_backward(self, module) -> Optional[torch.Tensor]:
        """Extract a tensor from the module after the backward pass.

        Called by the backbone after ``loss.backward()`` for each calibration
        batch, but only when ``requires_backward`` is ``True``. The default
        implementation returns ``None`` (no post-backward data). Override this
        for gradient-based collectors that read ``module.weight.grad`` or
        similar.

        Args:
            module: The ``LinearCompressed`` module whose gradients (or other
                post-backward state) should be extracted.

        Returns:
            A ``torch.Tensor`` with the collected data, or ``None`` if nothing
            should be emitted for this batch.
        """
        return None
