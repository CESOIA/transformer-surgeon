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

    name: str
    requires_backward: bool = False
    requires_loss: bool = False
    requires_labels: bool = False

    def build_forward_hook(
        self,
        *,
        emit_raw: Callable[[str, torch.Tensor], None],
    ):
        """Optional forward hook factory."""
        return None

    def collect_after_backward(self, module) -> Optional[torch.Tensor]:
        """Optional post-backward collection point."""
        return None


def normalize_calibration_batch(batch, batch_id: int):
    """Normalize dataloader output into model args/kwargs and optional explicit label."""
    if isinstance(batch, Mapping):
        return tuple(), dict(batch), None

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError(f"Calibration batch is empty at batch index {batch_id}.")

        if len(batch) == 1:
            item = batch[0]
            if isinstance(item, Mapping):
                return tuple(), dict(item), None
            return (item,), {}, None

        if len(batch) == 2 and isinstance(batch[0], Mapping):
            return tuple(), dict(batch[0]), batch[1]

        if len(batch) == 2 and isinstance(batch[1], Mapping):
            return tuple(), dict(batch[1]), batch[0]

        return tuple(batch), {}, None

    return (batch,), {}, None


def inject_labels_if_needed(model_args, model_kwargs, label):
    """Inject labels into kwargs for loss-producing calibration paths."""
    if isinstance(label, Mapping):
        for key, value in label.items():
            if key in model_kwargs:
                raise ValueError(f"Label key '{key}' already exists in calibration data mapping.")
            model_kwargs[key] = value
        return model_args, model_kwargs

    if any(key in model_kwargs for key in LABEL_KEYS):
        return model_args, model_kwargs

    if label is not None:
        model_kwargs["labels"] = label
        return model_args, model_kwargs

    if len(model_args) >= 2:
        model_kwargs["labels"] = model_args[-1]
        return model_args[:-1], model_kwargs

    raise ValueError(
        "Calibration path requiring labels could not infer them. "
        "Use batches with labels in kwargs, (data, label), or positional args ending with label."
    )
