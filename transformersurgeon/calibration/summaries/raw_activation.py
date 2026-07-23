"""Raw (unreduced) activation summaries.

Distinct from ActivationRangeSummary/OutputActivationRangeSummary (which only
keep a running min/max): these accumulate and concatenate the raw per-batch
activation tensors across the whole calibration pass.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class _BaseRawActivationSummary(CalibrationSummary):
    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        raw_name = self.required_raw_data[0]
        calibration_store.setdefault(self.name, []).append(raw_data[raw_name])

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        # Unused: update_runtime is overridden directly above.
        raise NotImplementedError

    def finalize_store(self, calibration_store: dict) -> None:
        if self.name in calibration_store:
            calibration_store[self.name] = torch.cat(calibration_store[self.name], dim=0)


class InputActivationSummary(_BaseRawActivationSummary):
    name = "input_activation"
    required_raw_data = ("activation",)


class OutputActivationSummary(_BaseRawActivationSummary):
    name = "output_activation"
    required_raw_data = ("output_activation",)
