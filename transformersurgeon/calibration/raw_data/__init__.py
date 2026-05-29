"""Raw data collection package exports."""

# Base contracts and shared batch-normalization helpers.
from .base import LABEL_KEYS, RawDataCollector, inject_labels_if_needed, normalize_calibration_batch
# Registry entrypoints used by the calibration backbone.
from .registry import SUPPORTED_RAW_DATA, instantiate_raw_collector

__all__ = [
    "LABEL_KEYS",
    "RawDataCollector",
    "SUPPORTED_RAW_DATA",
    "instantiate_raw_collector",
    "normalize_calibration_batch",
    "inject_labels_if_needed",
]
