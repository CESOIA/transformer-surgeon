"""
Base raw data contracts and shared calibration batch utilities.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
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

    requires_loss: bool = False
    """Reserved flag for collectors that need direct access to the scalar loss
    value. The current backbone does not auto-inject the loss; this flag is
    provided for future use."""

    requires_labels: bool = False
    """Reserved flag for collectors that depend on explicit target labels. The
    current backbone does not auto-inject targets into model ``kwargs``; this
    flag is provided for future use."""

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


def normalize_calibration_batch(batch, batch_id: int):
    """Normalize dataloader output into model args/kwargs and optional target object."""
    # Internal fast-path used by block-wise cascade flow.
    # Expected shape:
    # {
    #   "__ts_args__": tuple | list | single positional arg,
    #   "__ts_kwargs__": mapping of keyword args,
    #   "__ts_label__": optional target/label
    # }
    if isinstance(batch, Mapping) and "__ts_args__" in batch:
        # Read reserved pre-normalized fields with safe defaults.
        model_args = batch.get("__ts_args__", tuple())
        model_kwargs = batch.get("__ts_kwargs__", {})
        label = batch.get("__ts_label__", None)

        # Normalize positional args to tuple for a stable downstream call contract.
        if not isinstance(model_args, tuple):
            if isinstance(model_args, list):
                model_args = tuple(model_args)
            else:
                # Single positional value -> one-element args tuple.
                model_args = (model_args,)

        # kwargs must always be mapping-like (dict-compatible for **kwargs usage).
        if not isinstance(model_kwargs, Mapping):
            raise TypeError(
                "Pre-normalized calibration batch '__ts_kwargs__' must be a mapping. "
                f"Got {type(model_kwargs)} at batch index {batch_id}."
            )

        return model_args, dict(model_kwargs), label

    # Mapping batch => keyword-only model call.
    if isinstance(batch, Mapping):
        return tuple(), dict(batch), None

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError(f"Calibration batch is empty at batch index {batch_id}.")

        if len(batch) == 1:
            item = batch[0]
            # Preserve mapping semantics even when wrapped in a single-element tuple/list.
            if isinstance(item, Mapping):
                return tuple(), dict(item), None
            return (item,), {}, None

        # Canonical (data_mapping, target) form.
        if len(batch) == 2 and isinstance(batch[0], Mapping):
            return tuple(), dict(batch[0]), batch[1]

        # Also support swapped form (target, data_mapping).
        if len(batch) == 2 and isinstance(batch[1], Mapping):
            return tuple(), dict(batch[1]), batch[0]

        # Fallback: treat as pure positional model args.
        return tuple(batch), {}, None

    # Single object/tensor batch => one positional argument.
    return (batch,), {}, None
