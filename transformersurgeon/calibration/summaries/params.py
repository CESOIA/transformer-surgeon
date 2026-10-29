"""Weight/bias parameter summaries."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class _ParamSummary(CalibrationSummary):
    """Stores the parameter stream of the same name (constant across batches: last batch wins)."""

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        calibration_store[self.name] = raw_data[self.name]


class WeightSummary(_ParamSummary):
    name = "weight"
    required_raw_data = ("weight",)


class BiasSummary(_ParamSummary):
    name = "bias"
    required_raw_data = ("bias",)
