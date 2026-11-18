"""Per-input-channel mean activation summary (the FLAP baseline value)."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class InputMeanSummary(CalibrationSummary):
    name = "input_mean"
    # Computed from module input activations, shape [tokens, in_features].
    required_raw_data = ("activation",)

    @staticmethod
    def _batch_sum_and_tokens(raw_data: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        activation = raw_data["activation"].float()
        batch_tokens = activation.size(0)
        if batch_tokens <= 0:
            raise ValueError("Input mean calibration received an empty activation batch.")
        return activation.sum(0), batch_tokens

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        batch_sum, batch_tokens = self._batch_sum_and_tokens(raw_data)
        calibration_store[self.name] = batch_sum / batch_tokens

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        batch_sum, batch_tokens = self._batch_sum_and_tokens(raw_data)
        prev_tokens = runtime.state.get("token_count", 0)
        if prev_tokens == 0 or self.name not in calibration_store:
            calibration_store[self.name] = batch_sum / batch_tokens
        else:
            # Token-weighted running average, as in input_sq_mean and the covariance summaries.
            calibration_store[self.name] = (calibration_store[self.name] * prev_tokens + batch_sum) / (prev_tokens + batch_tokens)
        runtime.state["token_count"] = prev_tokens + batch_tokens
