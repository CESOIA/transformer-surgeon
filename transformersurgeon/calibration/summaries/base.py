"""Base summary contracts and runtime state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple

import torch


class CalibrationSummary(ABC):
    """Abstract summary contract used by the calibration backbone."""

    # Registry key used by compressors in needs_calibration().
    name: str
    # Raw streams required before one summary update can happen.
    required_raw_data: Tuple[str, ...]
    # If True, this summary requires a shifted model activation stream.
    requires_shifted_model: bool = False

    def initialize_store(self, calibration_store: dict) -> None:
        """Prepare storage keys owned by this summary."""
        # Reset only this summary namespace, keep other summaries untouched.
        calibration_store.pop(self.name, None)

    @abstractmethod
    def update_from_raw(self, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        """Update summary using a complete raw-data payload for one calibration step."""

    def update_runtime(self, runtime, calibration_store: dict, raw_data: Mapping[str, torch.Tensor]) -> None:
        """Runtime-aware update hook for summaries that need internal state."""
        # Default implementation ignores runtime state and performs a direct overwrite.
        self.update_from_raw(calibration_store, raw_data)

    def finalize_store(self, calibration_store: dict) -> None:
        """Finalize summary values after all calibration batches are processed."""


@dataclass
class SummaryRuntime:
    """Tracks per-batch pending raw inputs for one summary instance."""

    summary: CalibrationSummary
    pending_raw: Dict[str, list[torch.Tensor]] = field(default_factory=dict)
    num_updates: int = 0
    state: Dict[str, int] = field(default_factory=dict)

    def reset_batch(self) -> None:
        # Clear queues between batches; leftovers indicate stream pairing mismatch.
        leftovers = {
            raw_name: len(queue)
            for raw_name, queue in self.pending_raw.items()
            if len(queue) > 0
        }
        if len(leftovers) > 0:
            raise RuntimeError(
                "Calibration raw stream mismatch: unpaired raw payloads remained at batch boundary "
                f"for summary {self.summary.name}: {leftovers}."
            )
        self.pending_raw.clear()

    def on_raw(self, calibration_store: dict, raw_name: str, raw_value: torch.Tensor) -> None:
        # Ignore raw streams unrelated to this summary.
        if raw_name not in self.summary.required_raw_data:
            return

        self.pending_raw.setdefault(raw_name, []).append(raw_value)

        # Update while complete FIFO payloads are available across required streams.
        while all(len(self.pending_raw.get(name, [])) > 0 for name in self.summary.required_raw_data):
            payload = {
                name: self.pending_raw[name].pop(0)
                for name in self.summary.required_raw_data
            }
            self.summary.update_runtime(self, calibration_store, payload)
            self.num_updates += 1


def unique_summaries(summary_names: Iterable[str]) -> Tuple[str, ...]:
    # Order-preserving de-duplication for deterministic calibration behavior.
    unique = []
    for summary_name in summary_names:
        if summary_name not in unique:
            unique.append(summary_name)
    return tuple(unique)
