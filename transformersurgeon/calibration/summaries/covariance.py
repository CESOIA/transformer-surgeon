"""Covariance summary specialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .base import CalibrationSummary


class _BaseCovarianceSummary(CalibrationSummary):
    empty_batch_error_message = "Covariance calibration received an empty activation batch."

    def _compute_batch_sum_and_tokens(
        self, raw_data: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        raise NotImplementedError

    def _validated_batch_sum_and_tokens(
        self, raw_data: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        batch_sum, batch_tokens = self._compute_batch_sum_and_tokens(raw_data)
        if batch_tokens <= 0:
            raise ValueError(self.empty_batch_error_message)
        return batch_sum, batch_tokens

    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        batch_sum, batch_tokens = self._validated_batch_sum_and_tokens(raw_data)
        # Single-batch fallback: E[x^T x] estimated from current token chunk only.
        calibration_store[self.name] = batch_sum / batch_tokens

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        batch_sum, batch_tokens = self._validated_batch_sum_and_tokens(raw_data)
        prev_tokens = runtime.state.get("token_count", 0)
        if prev_tokens == 0 or self.name not in calibration_store:
            calibration_store[self.name] = batch_sum / batch_tokens
        else:
            # Weighted running average by token count avoids bias from variable batch sizes.
            total_tokens = prev_tokens + batch_tokens
            calibration_store[self.name] = (
                calibration_store[self.name] * prev_tokens + batch_sum
            ) / total_tokens
        # Persist cumulative token count for next update.
        runtime.state["token_count"] = prev_tokens + batch_tokens


class CovarianceSummary(_BaseCovarianceSummary):
    name = "covariance"
    # Covariance is computed from module input activations.
    required_raw_data = ("activation",)

    def _compute_batch_sum_and_tokens(
        self, raw_data: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        activation = raw_data["activation"].float()
        batch_tokens = activation.size(0)
        # Keep unnormalized X^T X for the incoming batch.
        batch_cov_sum = activation.transpose(0, 1) @ activation
        return batch_cov_sum, batch_tokens


class ShiftedCovarianceSummary(_BaseCovarianceSummary):
    name = "shifted_covariance"
    # Covariance over shifted-model activations only.
    required_raw_data = ("activation_shifted",)
    requires_shifted_model = True
    empty_batch_error_message = "Shifted covariance calibration received an empty activation batch."

    def _compute_batch_sum_and_tokens(
        self, raw_data: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        activation_shifted = raw_data["activation_shifted"].float()
        batch_tokens = activation_shifted.size(0)
        batch_cov_sum = activation_shifted.transpose(0, 1) @ activation_shifted
        return batch_cov_sum, batch_tokens


class CrossCovarianceSummary(_BaseCovarianceSummary):
    name = "cross_covariance"
    # Cross covariance between base-model and shifted-model activations.
    required_raw_data = ("activation", "activation_shifted")
    requires_shifted_model = True
    empty_batch_error_message = "Cross covariance calibration received an empty activation batch."

    def _compute_batch_sum_and_tokens(
        self, raw_data: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int]:
        activation = raw_data["activation"].float()
        activation_shifted = raw_data["activation_shifted"].float()
        if activation.size(0) != activation_shifted.size(0):
            raise ValueError(
                "Cross covariance calibration received activation batches with different token counts: "
                f"{activation.size(0)} vs {activation_shifted.size(0)}."
            )
        batch_tokens = activation.size(0)
        batch_cross_cov_sum = activation.transpose(0, 1) @ activation_shifted
        return batch_cross_cov_sum, batch_tokens
