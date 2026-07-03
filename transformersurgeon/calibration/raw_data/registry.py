"""Raw data collector registry."""

from __future__ import annotations

from .activation import ActivationCollector, ShiftedActivationCollector, OutputActivationCollector
from .weight_grad import WeightGradCollector


RAW_DATA_REGISTRY = {
    # Activation stream used by covariance-like summaries.
    "activation": ActivationCollector,
    # Activation stream collected from the shifted model.
    "activation_shifted": ShiftedActivationCollector,
    # Output activation stream used by output-side activation quantization.
    "output_activation": OutputActivationCollector,
    # Weight-gradient stream used by gradient-aware pruning methods.
    "weight_grad": WeightGradCollector,
}
SUPPORTED_RAW_DATA = tuple(RAW_DATA_REGISTRY.keys())


def instantiate_raw_collector(raw_name: str, *, offload_to_cpu: bool):
    # Factory used by the backbone to instantiate one collector per raw stream type.
    try:
        collector_type = RAW_DATA_REGISTRY[raw_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported calibration raw data '{raw_name}'. Supported raw data are: {SUPPORTED_RAW_DATA}."
        ) from exc
    return collector_type(offload_to_cpu=offload_to_cpu)
