"""Summary package exports."""

# Base contracts and runtime glue for summary updates.
from .base import CalibrationSummary, SummaryRuntime, unique_summaries
# Registry helpers for summary lookup by name.
from .registry import SUPPORTED_SUMMARIES, get_summary

__all__ = [
    "CalibrationSummary",
    "SummaryRuntime",
    "SUPPORTED_SUMMARIES",
    "get_summary",
    "unique_summaries",
]
