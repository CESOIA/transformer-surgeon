"""
Calibration backbone.

This runner is intentionally summary-agnostic: it only orchestrates model
execution and dispatches raw data to summary runtimes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .raw_data import inject_labels_if_needed, instantiate_raw_collector, normalize_calibration_batch
from .summaries import SummaryRuntime, get_summary, unique_summaries
from ..utils.utils import infer_model_device, move_to_device


class _FallbackProgressBar:
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


def _build_progress_bar(*, enabled: bool, total: Optional[int], verbose: bool):
    if not enabled:
        return _FallbackProgressBar(total=total, enabled=False)

    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, desc="Calibration", leave=False, disable=not verbose)
    except Exception:
        return _FallbackProgressBar(total=total, enabled=verbose)


def _extract_loss(model_output, label, loss_fn):
    if isinstance(model_output, torch.Tensor):
        loss = model_output
    elif hasattr(model_output, "loss"):
        loss = model_output.loss
    elif isinstance(model_output, Mapping) and "loss" in model_output:
        loss = model_output["loss"]
    else:
        loss = None

    if loss is None and loss_fn is not None:
        loss = loss_fn(model_output, label)

    if loss is None:
        return None

    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"Extracted loss must be a torch.Tensor, but got {type(loss)}.")
    if loss.numel() != 1:
        raise ValueError(f"Extracted loss must be scalar, but got shape {tuple(loss.shape)}.")
    if not loss.requires_grad:
        raise ValueError("Extracted loss does not require gradients.")
    if not torch.isfinite(loss.detach()).all():
        raise ValueError("Extracted loss is not finite (NaN/Inf) during calibration.")

    return loss


def _collect_targets(manager, criteria=None):
    targets = []
    for scheme in manager.iter_filtered(criteria=criteria):
        required_summaries = []

        for compressor in scheme.compressors.values():
            if not compressor._to_compress():
                continue

            compressor.set_calibration_store(scheme.calibration_data)

            needs_calibration_fn = getattr(compressor, "needs_calibration", None)
            if not callable(needs_calibration_fn):
                continue

            needs_calibration = needs_calibration_fn()
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
                if summary_name not in required_summaries:
                    required_summaries.append(summary_name)

        if len(required_summaries) > 0:
            targets.append((scheme, unique_summaries(required_summaries)))

    return targets


def run_compression_calibration(
    manager,
    criteria=None,
    loss_fn=None,
    max_batches: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    offload_to_cpu: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
):
    """
    Run calibration by routing raw data through summary runtimes.

    Expected calibration dataloader format:
    - Mapping kwargs batches
    - (data, label) where data is mapping
    - Positional tuple/list batches
    - Single tensor/object batches
    """
    calibration_data = manager.calibration_data
    if calibration_data is None:
        raise ValueError(
            "Calibration data is required before running calibration. "
            "Call manager.set_calibration_data(...) before manager.apply(...)."
        )
    if not isinstance(calibration_data, DataLoader):
        raise TypeError(
            "Calibration data must be a torch.utils.data.DataLoader. "
            f"Got {type(calibration_data)}."
        )

    targets = _collect_targets(manager, criteria=criteria)
    if len(targets) == 0:
        if verbose:
            print("No compression schemes found for calibration.")
        return 0

    model = manager.model
    module_by_scheme = {scheme: scheme.get_compression_module() for scheme, _ in targets}

    runtimes_by_scheme: Dict[object, Tuple[SummaryRuntime, ...]] = {}
    raw_required_by_scheme: Dict[object, set] = {}
    required_raw_names = set()

    for scheme, summary_names in targets:
        runtimes = []
        raw_names = set()
        for summary_name in summary_names:
            summary = get_summary(summary_name)
            summary.initialize_store(scheme.calibration_data)
            runtime = SummaryRuntime(summary=summary)
            runtimes.append(runtime)
            raw_names.update(summary.required_raw_data)
        runtimes_by_scheme[scheme] = tuple(runtimes)
        raw_required_by_scheme[scheme] = raw_names
        required_raw_names.update(raw_names)

    raw_collectors = {
        raw_name: instantiate_raw_collector(raw_name, offload_to_cpu=offload_to_cpu)
        for raw_name in required_raw_names
    }

    requires_backward = any(c.requires_backward for c in raw_collectors.values())
    requires_labels = any(c.requires_labels for c in raw_collectors.values())

    hook_handles = []

    def _dispatch_raw(scheme, raw_name: str, raw_value: torch.Tensor):
        for runtime in runtimes_by_scheme[scheme]:
            runtime.on_raw(scheme.calibration_data, raw_name, raw_value)

    for scheme, module in module_by_scheme.items():
        for raw_name in raw_required_by_scheme[scheme]:
            collector = raw_collectors[raw_name]
            hook = collector.build_forward_hook(
                emit_raw=lambda name, value, _scheme=scheme: _dispatch_raw(_scheme, name, value)
            )
            if hook is not None:
                hook_handles.append(module.register_forward_hook(hook))

    if device is None:
        device = infer_model_device(model)
    device = torch.device(device)

    was_training = model.training
    num_batches = 0
    total_batches = None
    try:
        total_batches = len(calibration_data)
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)
    except TypeError:
        if max_batches is not None:
            total_batches = max_batches

    progress_bar = _build_progress_bar(enabled=show_progress, total=total_batches, verbose=verbose)

    try:
        model.eval()

        for batch_id, batch in enumerate(calibration_data):
            if max_batches is not None and batch_id >= max_batches:
                break

            for runtimes in runtimes_by_scheme.values():
                for runtime in runtimes:
                    runtime.reset_batch()

            model_args, model_kwargs, label = normalize_calibration_batch(batch, batch_id=batch_id)
            model_args = move_to_device(model_args, device)
            model_kwargs = move_to_device(model_kwargs, device)
            label = move_to_device(label, device)

            model_args = tuple(model_args)
            model_kwargs = dict(model_kwargs)

            if requires_labels:
                model_args, model_kwargs = inject_labels_if_needed(model_args, model_kwargs, label)

            if requires_backward:
                model.zero_grad(set_to_none=True)
                model_output = model(*model_args, **model_kwargs)
                loss = _extract_loss(model_output, label, loss_fn)
                if loss is None:
                    raise ValueError(
                        "Calibration path requiring backward did not produce a loss value. "
                        f"No loss was found for batch {batch_id}."
                    )
                loss.backward()

                for scheme, module in module_by_scheme.items():
                    for raw_name in raw_required_by_scheme[scheme]:
                        collector = raw_collectors[raw_name]
                        raw_value = collector.collect_after_backward(module)
                        if raw_value is None:
                            continue
                        _dispatch_raw(scheme, raw_name, raw_value)
            else:
                with torch.no_grad():
                    model(*model_args, **model_kwargs)

            num_batches += 1
            progress_bar.update(1)

    finally:
        progress_bar.close()
        for handle in hook_handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    if num_batches == 0:
        raise ValueError("Calibration data must contain at least one batch.")

    for scheme, runtimes in runtimes_by_scheme.items():
        for runtime in runtimes:
            if runtime.num_updates == 0:
                raise RuntimeError(
                    f"No calibration raw data were collected for summary {runtime.summary.name} "
                    f"on module {scheme.path}."
                )
            runtime.summary.finalize_store(scheme.calibration_data)

    if verbose:
        print(f"Calibrated {len(targets)} schemes over {num_batches} batches.")

    return len(targets)


__all__ = ["run_compression_calibration"]
