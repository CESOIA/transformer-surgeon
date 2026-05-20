"""Weight gradient summary specialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class WeightGradSummary(CalibrationSummary):
    name = "weight_grad"
    required_raw_data = ("weight_grad",)

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        calibration_store[self.name] = raw_data["weight_grad"]

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        grad = raw_data["weight_grad"]
        prev_batches = runtime.state.get("num_batches", 0)
        if prev_batches == 0 or self.name not in calibration_store:
            calibration_store[self.name] = grad
        else:
            calibration_store[self.name] = (
                calibration_store[self.name] * prev_batches + grad
            ) / (prev_batches + 1)
        runtime.state["num_batches"] = prev_batches + 1
