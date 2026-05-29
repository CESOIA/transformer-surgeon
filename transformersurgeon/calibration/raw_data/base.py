"""
Base raw data contracts and shared calibration batch utilities.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Callable, Optional

import torch


LABEL_KEYS = ("labels", "label", "lm_labels", "start_positions", "end_positions")


class RawDataCollector(ABC):
    """Base collector used by the calibration backbone."""

    # Registry name used in summary.required_raw_data.
    name: str
    # If True, collector needs post-backward extraction.
    requires_backward: bool = False
    # Reserved capability flag for collectors requiring explicit loss access.
    requires_loss: bool = False
    # If True, caller must ensure labels are present in model kwargs.
    requires_labels: bool = False

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
    """Normalize dataloader output into model args/kwargs and optional explicit label."""
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

        # Canonical (data_mapping, label) form.
        if len(batch) == 2 and isinstance(batch[0], Mapping):
            return tuple(), dict(batch[0]), batch[1]

        # Also support swapped form (label, data_mapping).
        if len(batch) == 2 and isinstance(batch[1], Mapping):
            return tuple(), dict(batch[1]), batch[0]

        # Fallback: treat as pure positional model args.
        return tuple(batch), {}, None

    # Single object/tensor batch => one positional argument.
    return (batch,), {}, None


def inject_labels_if_needed(model_args, model_kwargs, label):
    """Inject labels into kwargs for loss-producing calibration paths."""
    # If label is already a mapping, merge every field into kwargs.
    if isinstance(label, Mapping):
        for key, value in label.items():
            if key in model_kwargs:
                raise ValueError(f"Label key '{key}' already exists in calibration data mapping.")
            model_kwargs[key] = value
        return model_args, model_kwargs

    # Respect datasets/collators that already expose known label keys.
    if any(key in model_kwargs for key in LABEL_KEYS):
        return model_args, model_kwargs

    # Direct label object from (data, label) loaders.
    if label is not None:
        model_kwargs["labels"] = label
        return model_args, model_kwargs

    # Last fallback for positional batches: assume last arg is labels.
    if len(model_args) >= 2:
        model_kwargs["labels"] = model_args[-1]
        return model_args[:-1], model_kwargs

    raise ValueError(
        "Calibration path requiring labels could not infer them. "
        "Use batches with labels in kwargs, (data, label), or positional args ending with label."
    )
