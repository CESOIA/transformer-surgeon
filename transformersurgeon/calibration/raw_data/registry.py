"""Raw data collector registry."""

from __future__ import annotations

from .activation import ActivationCollector
from .weight_grad import WeightGradCollector


RAW_DATA_REGISTRY = {
    "activation": ActivationCollector,
    "weight_grad": WeightGradCollector,
}
SUPPORTED_RAW_DATA = tuple(RAW_DATA_REGISTRY.keys())


def instantiate_raw_collector(raw_name: str, *, offload_to_cpu: bool):
    try:
        collector_type = RAW_DATA_REGISTRY[raw_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported calibration raw data '{raw_name}'. Supported raw data are: {SUPPORTED_RAW_DATA}."
        ) from exc
    return collector_type(offload_to_cpu=offload_to_cpu)
