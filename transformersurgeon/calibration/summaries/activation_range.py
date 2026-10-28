"""Activation range summaries for static activation quantization calibration."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class _RangeSummary(CalibrationSummary):
    """Running min/max of the single raw stream in ``required_raw_data``.

    Stored value: ``calibration_store[name] = {"min": tensor, "max": tensor}``.
    """

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        x = raw_data[self.required_raw_data[0]].float()
        batch_min, batch_max = x.min(), x.max()
        prev = calibration_store.get(self.name)
        if prev is None:
            calibration_store[self.name] = {"min": batch_min, "max": batch_max}
        else:
            calibration_store[self.name] = {"min": torch.min(prev["min"], batch_min),
                                            "max": torch.max(prev["max"], batch_max)}


class ActivationRangeSummary(_RangeSummary):
    """Range of the layer *input* activations (static input-activation quantisation)."""

    name = "activation_range"
    required_raw_data = ("activation",)


class OutputActivationRangeSummary(_RangeSummary):
    """Range of the layer *output* activations (static output-activation quantisation)."""

    name = "output_activation_range"
    required_raw_data = ("output_activation",)
