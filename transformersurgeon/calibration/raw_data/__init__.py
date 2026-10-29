"""Raw data collection package exports."""

# Base contracts and shared batch-normalization helpers.
from .base import RawDataCollector
# Registry entrypoints used by the calibration backbone.
from .registry import SUPPORTED_RAW_DATA, instantiate_raw_collector

__all__ = [
    "RawDataCollector",
    "SUPPORTED_RAW_DATA",
    "instantiate_raw_collector",
]
