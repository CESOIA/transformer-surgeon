"""Calibration framework package."""

# Public entrypoint used by the manager to run the full calibration pass.
from .backbone import run_compression_calibration
# Public list of summary names currently supported by the framework.
from .summaries import SUPPORTED_SUMMARIES

# Standalone pass: per-input-channel max |x| (used by SmoothQuant).
from .input_absmax import collect_input_absmax

# Calibration data: token windows from a corpus (WikiText-2 shortcut).
from .datasets import token_windows_loader, wikitext2_loader

__all__ = [
    "run_compression_calibration",
    "SUPPORTED_SUMMARIES",
    "collect_input_absmax",
    "token_windows_loader",
    "wikitext2_loader",
]
