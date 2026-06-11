"""Summary registry."""

from __future__ import annotations

from .activation_range import ActivationRangeSummary
from .covariance import CovarianceSummary, CrossCovarianceSummary, ShiftedCovarianceSummary
from .weight_grad import WeightGradSummary


SUMMARY_REGISTRY = {
    # Running activation covariance, used by calibrated LRD methods (e.g. svd-llm-v2).
    "covariance": CovarianceSummary(),
    # Running covariance over shifted-model activations.
    "shifted_covariance": ShiftedCovarianceSummary(),
    # Running cross covariance between base and shifted activations.
    "cross_covariance": CrossCovarianceSummary(),
    # Running weight-gradient average, used by gradient-based pruning methods.
    "weight_grad": WeightGradSummary(),
    # Running min/max of input activations, used by static activation fake-quant.
    "activation_range": ActivationRangeSummary(),
}
SUPPORTED_SUMMARIES = tuple(SUMMARY_REGISTRY.keys())


def get_summary(summary_name: str):
    # Name-to-instance lookup used by calibration backbone when wiring runtimes.
    try:
        return SUMMARY_REGISTRY[summary_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported calibration summary '{summary_name}'. Supported summaries are: {SUPPORTED_SUMMARIES}."
        ) from exc
