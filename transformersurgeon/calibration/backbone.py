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
from .summaries import SummaryRuntime, get_summary
from .targets import collect_targets
from ..utils.interface import build_progress_bar
from ..utils.utils import infer_model_device, move_to_device


def _extract_loss(model_output, label, loss_fn):
    # Accept different model return conventions: tensor, dataclass-like, or mapping.
    if isinstance(model_output, torch.Tensor):
        loss = model_output
    elif hasattr(model_output, "loss"):
        loss = model_output.loss
    elif isinstance(model_output, Mapping) and "loss" in model_output:
        loss = model_output["loss"]
    else:
        loss = None

    if loss is None and loss_fn is not None:
        # Allow callers to define a custom extraction rule when model output is non-standard.
        loss = loss_fn(model_output, label)

    if loss is None:
        return None

    if not isinstance(loss, torch.Tensor):
        raise TypeError(f"Extracted loss must be a torch.Tensor, but got {type(loss)}.")
    # Backward-based collectors require a scalar differentiable finite loss.
    if loss.numel() != 1:
        raise ValueError(f"Extracted loss must be scalar, but got shape {tuple(loss.shape)}.")
    if not loss.requires_grad:
        raise ValueError("Extracted loss does not require gradients.")
    if not torch.isfinite(loss.detach()).all():
        raise ValueError("Extracted loss is not finite (NaN/Inf) during calibration.")

    return loss


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
    # Validate calibration input source first: this runner expects a DataLoader.
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

    # Discover only the schemes/summaries needed by active compressors.
    targets = collect_targets(manager, criteria=criteria)
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
        # Build one runtime per requested summary and accumulate required raw streams.
        runtimes = []
        raw_names = set()
        for summary_name in summary_names:
            summary = get_summary(summary_name)
            # Summary owns only its own key in this scheme's calibration store.
            summary.initialize_store(scheme.calibration_data)
            runtime = SummaryRuntime(summary=summary)
            runtimes.append(runtime)
            raw_names.update(summary.required_raw_data)
        runtimes_by_scheme[scheme] = tuple(runtimes)
        raw_required_by_scheme[scheme] = raw_names
        required_raw_names.update(raw_names)

    # Instantiate raw collectors once per raw stream type (shared across schemes).
    raw_collectors = {
        raw_name: instantiate_raw_collector(raw_name, offload_to_cpu=offload_to_cpu)
        for raw_name in required_raw_names
    }

    # Global execution mode flags drive forward-only vs forward+backward calibration.
    requires_backward = any(c.requires_backward for c in raw_collectors.values())
    requires_labels = any(c.requires_labels for c in raw_collectors.values())

    hook_handles = []

    def _dispatch_raw(scheme, raw_name: str, raw_value: torch.Tensor):
        # Fan out each raw tensor to all summary runtimes attached to that scheme.
        for runtime in runtimes_by_scheme[scheme]:
            runtime.on_raw(scheme.calibration_data, raw_name, raw_value)

    for scheme, module in module_by_scheme.items():
        for raw_name in raw_required_by_scheme[scheme]:
            collector = raw_collectors[raw_name]
            # Collectors expose optional forward hooks; hook output is always routed by raw name.
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

    progress_bar = build_progress_bar(enabled=show_progress, total=total_batches, verbose=verbose)

    try:
        model.eval()

        for batch_id, batch in enumerate(calibration_data):
            if max_batches is not None and batch_id >= max_batches:
                break

            # Reset per-batch runtime cache so each update uses only current-batch raw values.
            for runtimes in runtimes_by_scheme.values():
                for runtime in runtimes:
                    runtime.reset_batch()

            # Normalize heterogeneous loader outputs to a canonical model invocation form.
            model_args, model_kwargs, label = normalize_calibration_batch(batch, batch_id=batch_id)
            model_args = move_to_device(model_args, device)
            model_kwargs = move_to_device(model_kwargs, device)
            label = move_to_device(label, device)

            model_args = tuple(model_args)
            model_kwargs = dict(model_kwargs)

            if requires_labels:
                # Backward collectors that rely on loss require labels to be injected consistently.
                model_args, model_kwargs = inject_labels_if_needed(model_args, model_kwargs, label)

            if requires_backward:
                # Backward path: run forward, extract scalar loss, backprop, then pull post-backward raw values.
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
                # Forward-only path for summaries based solely on activations.
                with torch.no_grad():
                    model(*model_args, **model_kwargs)

            num_batches += 1
            progress_bar.update(1)

    finally:
        # Always clean hooks and restore training mode, even on failure.
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
            # Guard against silent failures where hooks never emitted required raw data.
            if runtime.num_updates == 0:
                raise RuntimeError(
                    f"No calibration raw data were collected for summary {runtime.summary.name} "
                    f"on module {scheme.path}."
                )
            # Finalization hook allows summaries to post-process at end of all batches.
            runtime.summary.finalize_store(scheme.calibration_data)

    if verbose:
        print(f"Calibrated {len(targets)} schemes over {num_batches} batches.")

    return len(targets)


__all__ = ["run_compression_calibration"]
