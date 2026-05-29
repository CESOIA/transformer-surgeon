"""Summary registry."""

from __future__ import annotations

from .covariance import CovarianceSummary
from .weight_grad import WeightGradSummary


SUMMARY_REGISTRY = {
    # Running activation covariance, used by calibrated LRD methods (e.g. svd-llm-v2).
    "covariance": CovarianceSummary(),
    # Running weight-gradient average, used by gradient-based pruning methods.
    "weight_grad": WeightGradSummary(),
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
