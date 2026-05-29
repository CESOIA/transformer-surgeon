"""Target discovery helpers for calibration."""

from __future__ import annotations

from .summaries import unique_summaries


def collect_targets(manager, criteria=None):
    """Return scheme targets and required summaries for calibration.

    Each returned item is a tuple: (scheme, summary_names).
    """
    # Build the list of (scheme, required summaries) that actually need calibration.
    targets = []
    for scheme in manager.iter_filtered(criteria=criteria):
        required_summaries = []

        for compressor in scheme.compressors.values():
            if not compressor._to_compress():
                continue

            # Give compressors access to scheme-local calibration storage.
            compressor.set_calibration_store(scheme.calibration_data)

            needs_calibration_fn = getattr(compressor, "needs_calibration", None)
            if not callable(needs_calibration_fn):
                continue

            needs_calibration = needs_calibration_fn()
            # False/None means "no calibration needed for current config".
            if needs_calibration is False or needs_calibration is None:
                continue

            if needs_calibration is True:
                raise ValueError(
                    f"Compressor {type(compressor).__name__} needs calibration but did not provide summary names."
                )

            if isinstance(needs_calibration, str):
                summary_names = (needs_calibration,)
            elif isinstance(needs_calibration, (tuple, list, set)):
                summary_names = tuple(needs_calibration)
            else:
                raise TypeError(
                    "needs_calibration must return False/None or a string/sequence of summary names."
                )

            if len(summary_names) == 0:
                raise ValueError(
                    f"Compressor {type(compressor).__name__} needs calibration but returned no summaries."
                )

            for summary_name in summary_names:
                # Preserve order while de-duplicating per scheme.
                if summary_name not in required_summaries:
                    required_summaries.append(summary_name)

        if len(required_summaries) > 0:
            targets.append((scheme, unique_summaries(required_summaries)))

    return targets


__all__ = ["collect_targets"]
