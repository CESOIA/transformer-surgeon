"""User-interface utilities shared across runtime components."""

from __future__ import annotations

from typing import Optional


class FallbackProgressBar:
    """Minimal in-place progress bar used when tqdm is unavailable."""

    def __init__(self, *, total: Optional[int], enabled: bool):
        self.total = total
        self.enabled = enabled
        self.current = 0

    def update(self, n: int = 1):
        if not self.enabled:
            return
        self.current += n
        if self.total is None or self.total <= 0:
            print(f"\rCalibration batches: {self.current}", end="", flush=True)
        else:
            pct = int((100 * self.current) / self.total)
            print(
                f"\rCalibration: {self.current}/{self.total} ({pct}%)",
                end="",
                flush=True,
            )

    def close(self):
        if self.enabled:
            print("", flush=True)


def build_progress_bar(*, enabled: bool, total: Optional[int], verbose: bool):
    """Return a tqdm progress bar when available, or a lightweight fallback."""
    if not enabled:
        return FallbackProgressBar(total=total, enabled=False)

    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc="Calibration", leave=False, disable=not verbose)
    except Exception:
        return FallbackProgressBar(total=total, enabled=verbose)


__all__ = ["FallbackProgressBar", "build_progress_bar"]
