"""
Base raw data contracts and shared calibration batch utilities.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Callable, Optional

import torch


class RawDataCollector(ABC):
    """Base collector used by the calibration backbone."""

    # Registry name used in summary.required_raw_data.
    name: str
    # If True, collector needs post-backward extraction.
    requires_backward: bool = False
    # Reserved capability flag for collectors requiring explicit loss access.
    requires_loss: bool = False
    # Reserved capability flag for collectors that depend on explicit targets.
    # The current backbone does not auto-inject targets into model kwargs.
    requires_labels: bool = False
    # If True, collector hook must be attached to shifted model modules.
    uses_shifted_model: bool = False

    def build_forward_hook(
        self,
        *,
        emit_raw: Callable[[str, torch.Tensor], None],
    ):
        """Optional forward hook factory."""
        # Return None when collector does not use forward-hook collection.
        return None

    def collect_after_backward(self, module) -> Optional[torch.Tensor]:
        """Optional post-backward collection point."""
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

