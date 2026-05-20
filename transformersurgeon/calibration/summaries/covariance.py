"""Covariance summary specialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class CovarianceSummary(CalibrationSummary):
    name = "covariance"
    required_raw_data = ("activation",)

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        activation = raw_data["activation"]
        batch_tokens = activation.size(0)
        if batch_tokens <= 0:
            raise ValueError("Covariance calibration received an empty activation batch.")

        calibration_store[self.name] = activation.transpose(0, 1) @ activation / batch_tokens

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        activation = raw_data["activation"]
        batch_tokens = activation.size(0)
        if batch_tokens <= 0:
            raise ValueError("Covariance calibration received an empty activation batch.")

        batch_cov_sum = activation.transpose(0, 1) @ activation
        prev_tokens = runtime.state.get("token_count", 0)
        if prev_tokens == 0 or self.name not in calibration_store:
            calibration_store[self.name] = batch_cov_sum / batch_tokens
        else:
            total_tokens = prev_tokens + batch_tokens
            calibration_store[self.name] = (
                calibration_store[self.name] * prev_tokens + batch_cov_sum
            ) / total_tokens
        runtime.state["token_count"] = prev_tokens + batch_tokens
