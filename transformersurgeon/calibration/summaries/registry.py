"""Summary registry."""

from __future__ import annotations

from .covariance import CovarianceSummary
from .weight_grad import WeightGradSummary


SUMMARY_REGISTRY = {
    "covariance": CovarianceSummary(),
    "weight_grad": WeightGradSummary(),
}
SUPPORTED_SUMMARIES = tuple(SUMMARY_REGISTRY.keys())


def get_summary(summary_name: str):
    try:
        return SUMMARY_REGISTRY[summary_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported calibration summary '{summary_name}'. Supported summaries are: {SUPPORTED_SUMMARIES}."
        ) from exc
