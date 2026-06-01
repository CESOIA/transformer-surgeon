"""Weight gradient summary specialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class WeightGradSummary(CalibrationSummary):
    name = "weight_grad"
    # Consumes gradients emitted by WeightGradCollector.
    required_raw_data = ("weight_grad",)

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        # Single-step behavior: store latest batch gradient as-is.
        calibration_store[self.name] = raw_data["weight_grad"]

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        grad = raw_data["weight_grad"]
        prev_batches = runtime.state.get("num_batches", 0)
        if prev_batches == 0 or self.name not in calibration_store:
            calibration_store[self.name] = grad
        else:
            # Running average over batches for a smoother gradient estimate.
            calibration_store[self.name] = (
                calibration_store[self.name] * prev_batches + grad
            ) / (prev_batches + 1)
        # Persist batch counter for the next weighted update.
        runtime.state["num_batches"] = prev_batches + 1
