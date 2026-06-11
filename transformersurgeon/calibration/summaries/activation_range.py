"""Activation range summary for static activation quantization calibration."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class ActivationRangeSummary(CalibrationSummary):
    """Tracks running min/max of input activations over a calibration dataset.

    Stored value: calibration_store["activation_range"] = {"min": tensor, "max": tensor}
    These are consumed by Quantizer.apply() to compute scale/zero_point for
    static activation fake-quantization emulation.
    """

    name = "activation_range"
    required_raw_data = ("activation",)

    def initialize_store(self, calibration_store: dict) -> None:
        calibration_store.pop(self.name, None)

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        x = raw_data["activation"].float()
        batch_min = x.min()
        batch_max = x.max()
        if self.name not in calibration_store:
            calibration_store[self.name] = {"min": batch_min, "max": batch_max}
        else:
            prev = calibration_store[self.name]
            calibration_store[self.name] = {
                "min": torch.min(prev["min"], batch_min),
                "max": torch.max(prev["max"], batch_max),
            }

    # update_runtime falls back to update_from_raw (base class default),
    # which is correct — running min/max is stateless across batches.
