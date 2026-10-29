"""Progress bar helper: ``tqdm`` when available, a minimal printer otherwise."""

from __future__ import annotations

from typing import Optional


class FallbackProgressBar:
    """Minimal in-place progress bar used when tqdm is unavailable."""

    def __init__(self, *, total: Optional[int], enabled: bool, desc: str = "Progress"):
        self.total = total
        self.enabled = enabled
        self.desc = desc
        self.current = 0

    def update(self, n: int = 1):
        if not self.enabled:
            return
        self.current += n
        if self.total is None or self.total <= 0:
            print(f"\r{self.desc}: {self.current}", end="", flush=True)
        else:
            pct = int((100 * self.current) / self.total)
            print(f"\r{self.desc}: {self.current}/{self.total} ({pct}%)", end="", flush=True)

    def close(self):
        if self.enabled:
            print("", flush=True)


def build_progress_bar(*, enabled: bool, total: Optional[int], verbose: bool, desc: str = "Progress"):
    """Return a tqdm progress bar when available, or a lightweight fallback."""
    if not enabled:
        return FallbackProgressBar(total=total, enabled=False, desc=desc)

    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, leave=False, disable=not verbose)
    except Exception:
        return FallbackProgressBar(total=total, enabled=verbose, desc=desc)


__all__ = ["FallbackProgressBar", "build_progress_bar"]
