"""
Calibration backbone.

This runner is intentionally summary-agnostic: it only orchestrates model
execution and dispatches raw data to summary runtimes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from .raw_data import instantiate_raw_collector, normalize_calibration_batch
from .summaries import SummaryRuntime, get_summary
from ..utils.interface import build_progress_bar
from ..utils.utils import get_submodule, infer_model_device, move_to_device


def _compute_calibration_loss(loss_fn, model_output, label):
    """Compute calibration loss via the user-provided callback."""
    if loss_fn is None:
        return None
    return loss_fn(model_output, label)


def run_compression_calibration(
    model,
    calibration_data,
    target_stages,
    shifted_model=None,
    loss_fn=None,
    max_batches: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    offload_to_cpu: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
):
    """
    Run calibration by routing raw data through summary runtimes.

    For backward-based collectors, loss_fn must be provided and must return
    a differentiable loss tensor from (model_output, label).

    Expected calibration dataloader format:
    - Mapping kwargs batches
    - (data, target) where data is mapping
    - Positional tuple/list batches
    - Single tensor/object batches
    """
    # Validate calibration input source first: this runner expects a DataLoader.
    if calibration_data is None:
        raise ValueError(
            "Calibration data is required before running calibration. "
            "Call manager.set_calibration_data(...) before manager.apply(...) or manager.run_calibration(...)."
        )
    if not isinstance(calibration_data, DataLoader):
        raise TypeError(
            "Calibration data must be a torch.utils.data.DataLoader. "
            f"Got {type(calibration_data)}."
        )

    if len(target_stages) == 0:
        if verbose:
            print("No compression schemes found for calibration.")
        return 0

    if device is None:
        device = infer_model_device(model)
    device = torch.device(device)

    shifted_device = None
    if shifted_model is not None:
        shifted_device = torch.device(infer_model_device(shifted_model))

    was_training = model.training
    shifted_was_training = shifted_model.training if shifted_model is not None else None
    calibrated_schemes = set()

    total_stages = len(target_stages)
    for stage_idx, targets in enumerate(target_stages, start=1):
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

        # Instantiate raw collectors once per raw stream type (shared across schemes in stage).
        raw_collectors = {
            raw_name: instantiate_raw_collector(raw_name, offload_to_cpu=offload_to_cpu)
            for raw_name in required_raw_names
        }

        requires_shifted_model = any(
            runtime.summary.requires_shifted_model
            for runtimes in runtimes_by_scheme.values()
            for runtime in runtimes
        )
        if requires_shifted_model and shifted_model is None:
            raise ValueError(
                "Selected calibration summaries require shifted_model. "
                "Pass shifted_model=... to manager.run_calibration(...) or manager.apply(...)."
            )

        shifted_module_by_scheme = {}
        if requires_shifted_model:
            for scheme in module_by_scheme:
                shifted_module = get_submodule(shifted_model, scheme.path)
                if hasattr(shifted_module, "block_b"):
                    shifted_module = shifted_module.block_b
                shifted_module_by_scheme[scheme] = shifted_module

        # Global execution mode flags drive forward-only vs forward+backward calibration.
        requires_backward = any(c.requires_backward for c in raw_collectors.values())
        if requires_backward and loss_fn is None:
            raise ValueError(
                "Calibration path requiring backward needs a calibration loss function. "
                "Call manager.set_calibration_loss(...) before manager.apply(...) or manager.run_calibration(...)."
            )

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
                    hook_module = shifted_module_by_scheme.get(scheme, module) if collector.uses_shifted_model else module
                    hook_handles.append(hook_module.register_forward_hook(hook))

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
            if requires_shifted_model:
                shifted_model.eval()

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
                shifted_model_args = model_args
                shifted_model_kwargs = model_kwargs
                if requires_shifted_model and shifted_device is not None:
                    shifted_model_args = tuple(move_to_device(model_args, shifted_device))
                    shifted_model_kwargs = dict(move_to_device(model_kwargs, shifted_device))

                if requires_backward:
                    # Backward path: run forward, compute user-provided loss, backprop, then pull post-backward raw values.
                    model.zero_grad(set_to_none=True)
                    model_output = model(*model_args, **model_kwargs)
                    loss = _compute_calibration_loss(loss_fn, model_output, label)
                    if loss is None:
                        raise ValueError(
                            "Calibration path requiring backward did not produce a loss value. "
                            f"Loss function returned None for batch {batch_id}."
                        )
                    loss.backward()

                    if requires_shifted_model:
                        # Shifted model is used for activation summaries only.
                        with torch.no_grad():
                            shifted_model(*shifted_model_args, **shifted_model_kwargs)

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
                        if requires_shifted_model:
                            shifted_model(*shifted_model_args, **shifted_model_kwargs)

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
            if requires_shifted_model and shifted_was_training:
                shifted_model.train()

        if num_batches == 0:
            raise ValueError("Calibration data must contain at least one batch.")

        for scheme, runtimes in runtimes_by_scheme.items():
            calibrated_schemes.add(scheme)
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
            print(
                f"Calibrated stage {stage_idx}/{total_stages}: "
                f"{len(targets)} schemes over {num_batches} batches."
            )

    return len(calibrated_schemes)


__all__ = ["run_compression_calibration"]
