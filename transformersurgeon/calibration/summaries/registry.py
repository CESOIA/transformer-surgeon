"""Summary registry."""

from __future__ import annotations

from .activation_range import ActivationRangeSummary, LowRankHiddenRangeSummary, OutputActivationRangeSummary
from .covariance import CovarianceSummary, CrossCovarianceSummary, ShiftedCovarianceSummary
from .input_sq_mean import InputSqMeanSummary
from .input_mean import InputMeanSummary
from .flap_scores import FlapScoresSummary, MagnitudeScoresSummary
from .weight_grad import WeightGradSummary
from .params import WeightSummary, BiasSummary
from .raw_activation import InputActivationSummary, OutputActivationSummary


SUMMARY_REGISTRY = {
    # Running activation covariance, used by calibrated LRD methods (e.g. svd-llm-v2).
    "covariance": CovarianceSummary(),
    # Running covariance over shifted-model activations.
    "shifted_covariance": ShiftedCovarianceSummary(),
    # Running cross covariance between base and shifted activations.
    "cross_covariance": CrossCovarianceSummary(),
    # Running per-input-channel mean of x^2, used by activation-aware pruning (wanda).
    "input_sq_mean": InputSqMeanSummary(),
    # Running per-input-channel mean of x (the baseline value of FLAP's bias compensation).
    "input_mean": InputMeanSummary(),
    # Running weight-gradient average, used by gradient-based pruning methods.
    "weight_grad": WeightGradSummary(),
    # Running min/max of input activations, used by static activation fake-quant.
    "activation_range": ActivationRangeSummary(),
    # Running min/max of output activations, used by output-side static fake-quant.
    "output_activation_range": OutputActivationRangeSummary(),
    # Running min/max of the inner activation h = V x of low-rank layers (static quantisation of U's input).
    "lrd_hidden_activation_range": LowRankHiddenRangeSummary(),
    # Raw module weight parameter, captured via the generic calibration forward-hook pipeline.
    "weight": WeightSummary(),
    # Raw module bias parameter, captured via the generic calibration forward-hook pipeline.
    "bias": BiasSummary(),
    # Concatenated raw input activations over the whole calibration pass.
    "input_activation": InputActivationSummary(),
    # Concatenated raw output activations over the whole calibration pass.
    "output_activation": OutputActivationSummary(),
}
SUPPORTED_SUMMARIES = tuple(SUMMARY_REGISTRY.keys())

# Global summaries: computed once across schemes after the calibration pass (see global_base.py).
GLOBAL_SUMMARY_REGISTRY = {
    # FLAP: standardised fluctuation scores per layer and one global threshold (keep-mask per layer).
    "flap_scores": FlapScoresSummary(),
    # Uncalibrated structured pruning: same global cut as FLAP, scores from the weight norms only.
    "magnitude_scores": MagnitudeScoresSummary(),
}
SUPPORTED_GLOBAL_SUMMARIES = tuple(GLOBAL_SUMMARY_REGISTRY.keys())


def get_global_summary(name: str):
    try:
        return GLOBAL_SUMMARY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported global calibration summary '{name}'. Supported: {SUPPORTED_GLOBAL_SUMMARIES}."
        ) from exc


def get_summary(summary_name: str):
    # Name-to-instance lookup used by calibration backbone when wiring runtimes.
    try:
        return SUMMARY_REGISTRY[summary_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported calibration summary '{summary_name}'. Supported summaries are: {SUPPORTED_SUMMARIES}."
        ) from exc
