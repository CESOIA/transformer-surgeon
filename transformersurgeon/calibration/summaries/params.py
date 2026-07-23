"""Weight/bias parameter summaries."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class WeightSummary(CalibrationSummary):
    name = "weight"
    required_raw_data = ("weight",)

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        # Constant across batches -- last-batch overwrite is fine.
        calibration_store[self.name] = raw_data["weight"]


class BiasSummary(CalibrationSummary):
    name = "bias"
    required_raw_data = ("bias",)

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        calibration_store[self.name] = raw_data["bias"]
