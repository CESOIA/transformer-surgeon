"""
Block-wise cascade calibration/apply orchestration.

This module intentionally keeps cascade logic out of manager.py so the main manager
remains compact and easier to maintain.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from ..calibration import run_compression_calibration
from ..calibration.raw_data import normalize_calibration_batch
from .utils import get_submodule, infer_model_device, move_to_device


def apply_cascade(
    manager,
    hard: bool = False,
    criteria=None,
    verbose: bool = False,
    max_batches: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    offload_to_cpu: bool = True,
    show_progress: bool = True,
):
    """
    Apply compression in block-wise cascade mode.

    High-level flow:
    1) Build ordered per-block layer stages from path_list hierarchy.
    2) For each block id, calibrate/apply stage by stage (layer groups).
    3) Calibration runs on a cloned uncompressed block; shifted model is the
       progressively compressed in-place block.
    4) After each block id, propagate block outputs to feed the next block id.
    """
    selected_schemes = list(manager.iter_filtered(criteria=criteria))
    if len(selected_schemes) == 0:
        return

    if manager.calibration_data is None:
        raise ValueError(
            "Cascade mode requires calibration data for block-wise inference. "
            "Call manager.set_calibration_data(...) before manager.apply(...)."
        )

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device(infer_model_device(manager.model))
    )

    model_was_training = manager.model.training
    manager.model.eval()

    try:
        for block_name, block_indexing in manager.indexing.items():
            block_scheme_dict = manager.schemes.get(block_name, {})
            block_scheme_set = set(block_scheme_dict.values())
            block_selected = [s for s in selected_schemes if s in block_scheme_set]
            if len(block_selected) == 0:
                continue

            path_entries, ordered_subblocks = _extract_path_entries(block_indexing["path_list"])
            path_template = block_indexing["path_template"]
            preprocessing_path = block_indexing.get("preprocessing", None)

            if preprocessing_path:
                preprocessing_module = get_submodule(manager.model, preprocessing_path)
                cached_block_inputs = _collect_preprocessing_outputs(
                    preprocessing_module=preprocessing_module,
                    calibration_data=manager.calibration_data,
                    max_batches=max_batches,
                    device=target_device,
                    offload_to_cpu=offload_to_cpu,
                )
            else:
                cached_block_inputs = _collect_loader_inputs(
                    calibration_data=manager.calibration_data,
                    max_batches=max_batches,
                    device=target_device,
                    offload_to_cpu=offload_to_cpu,
                )

            # Optional model-specific positional embedding injection.
            cached_block_inputs = _attach_position_embeddings_if_configured(
                cached_inputs=cached_block_inputs,
                block_indexing=block_indexing,
                model=manager.model,
                device=target_device,
                offload_to_cpu=offload_to_cpu,
            )

            block_ids = sorted({s.block_id for s in block_selected})
            selected_set = set(block_selected)

            for block_id in block_ids:
                stage_layers = _build_layer_stages_for_block(
                    manager=manager,
                    block_id=block_id,
                    selected_set=selected_set,
                    block_scheme_dict=block_scheme_dict,
                    path_entries=path_entries,
                    ordered_subblocks=ordered_subblocks,
                    block_name=block_name,
                )

                if len(stage_layers) == 0:
                    continue

                block_module = _resolve_block_container_module(
                    model=manager.model,
                    path_template=path_template,
                    block_id=block_id,
                )
                reference_block = _clone_module(block_module)

                touched_schemes = []

                for stage_idx, layer_group in enumerate(stage_layers, start=1):
                    stage_schemes = _stage_schemes_from_layer_group(
                        block_scheme_dict=block_scheme_dict,
                        path_template=path_template,
                        block_id=block_id,
                        layer_group=layer_group,
                    )

                    if len(stage_schemes) == 0:
                        continue

                    for scheme in stage_schemes:
                        if scheme not in touched_schemes:
                            touched_schemes.append(scheme)

                    calibration_targets = manager._collect_calibration_targets_for_schemes(stage_schemes)
                    if len(calibration_targets) > 0:
                        localized_targets = _localize_targets_to_block(
                            calibration_targets=calibration_targets,
                            path_template=path_template,
                            block_id=block_id,
                            block_model=reference_block,
                        )
                        cached_loader = DataLoader(cached_block_inputs, batch_size=None, shuffle=False)

                        run_compression_calibration(
                            model=reference_block,
                            calibration_data=cached_loader,
                            target_stages=[localized_targets],
                            shifted_model=block_module,
                            loss_fn=manager.calibration_loss_fn,
                            max_batches=max_batches,
                            device=target_device,
                            offload_to_cpu=offload_to_cpu,
                            verbose=verbose,
                            show_progress=show_progress,
                        )

                    for scheme in stage_schemes:
                        scheme.apply(hard=hard, verbose=verbose)

                    if verbose:
                        print(
                            f"Cascade block '{block_name}' id {block_id} stage "
                            f"{stage_idx}/{len(stage_layers)} applied on {len(stage_schemes)} scheme(s)."
                        )

                # Block outputs feed the next block id.
                cached_block_inputs = _run_block_on_cached_inputs(
                    block_module=block_module,
                    cached_inputs=cached_block_inputs,
                    device=target_device,
                    offload_to_cpu=offload_to_cpu,
                )

                # Recompute/inject position embeddings for the next block.
                # _run_block_on_cached_inputs stores clean kwargs, so any
                # model-specific positional inputs must be reattached here.
                cached_block_inputs = _attach_position_embeddings_if_configured(
                    cached_inputs=cached_block_inputs,
                    block_indexing=block_indexing,
                    model=manager.model,
                    device=target_device,
                    offload_to_cpu=offload_to_cpu,
                )

                # Free block-local calibration artifacts.
                for scheme in touched_schemes:
                    scheme.calibration_data.clear()
                del reference_block
                del touched_schemes

    finally:
        if model_was_training:
            manager.model.train()


def _extract_path_entries(path_list) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Convert path_list to ordered subblock -> layer-path mapping.

    Example dict form:
      {
        "self_attn": ["q_proj", "k_proj"],
        "input_layernorm": [],
      }

    becomes:
      {
        "self_attn": ["self_attn.q_proj", "self_attn.k_proj"],
        "input_layernorm": ["input_layernorm"],
      }
    """
    if isinstance(path_list, Mapping):
        ordered_subblocks = list(path_list.keys())
        entries: Dict[str, List[str]] = {}

        for subblock_name, child_layers in path_list.items():
            if not isinstance(child_layers, (list, tuple)):
                raise TypeError(
                    "Grouped path_list values must be a list/tuple of sublayers. "
                    f"Found {type(child_layers).__name__} for key '{subblock_name}'."
                )

            if len(child_layers) == 0:
                entries[subblock_name] = [subblock_name]
                continue

            expanded_layers = []
            for layer_name in child_layers:
                layer_name = str(layer_name)
                if subblock_name:
                    expanded_layers.append(f"{subblock_name}.{layer_name}")
                else:
                    expanded_layers.append(layer_name)
            entries[subblock_name] = expanded_layers

        return entries, ordered_subblocks

    if isinstance(path_list, (list, tuple)):
        as_list = [str(p) for p in path_list]
        return {p: [p] for p in as_list}, as_list

    raise TypeError(
        "Indexing 'path_list' must be a list/tuple or grouped dict, "
        f"got {type(path_list).__name__}."
    )


def _build_layer_stages_for_block(
    manager,
    block_id: int,
    selected_set: set,
    block_scheme_dict,
    path_entries: Dict[str, List[str]],
    ordered_subblocks: List[str],
    block_name: str,
) -> List[List[str]]:
    """
    Build ordered stage layers for one block id.

    Stages are layer-level groups. User groups are validated as consecutive in the
    flattened layer order (subblock order, then layer order within each subblock).
    """
    ordered_layers = []
    for subblock_name in ordered_subblocks:
        ordered_layers.extend(path_entries[subblock_name])

    available_layers = []
    for layer_path in ordered_layers:
        has_selected = False
        for scheme in block_scheme_dict.values():
            if scheme not in selected_set:
                continue
            if scheme.block_id != block_id:
                continue
            if scheme.name == layer_path:
                has_selected = True
                break
        if has_selected:
            available_layers.append(layer_path)

    if len(available_layers) == 0:
        return []

    if len(manager.calibration_groups) == 0:
        return [[layer] for layer in available_layers]

    grouped_stages: List[List[str]] = []
    grouped_set = set()
    index_by_layer = {layer: idx for idx, layer in enumerate(available_layers)}

    for group_idx, group_criteria in enumerate(manager.calibration_groups):
        matched_layers = [
            layer for layer in available_layers if _criteria_match_layer(layer, block_id, group_criteria)
        ]

        if len(matched_layers) == 0:
            continue

        overlap = [layer for layer in matched_layers if layer in grouped_set]
        if len(overlap) > 0:
            raise ValueError(
                f"Cascade calibration groups overlap on block '{block_name}' id {block_id}. "
                f"Group index {group_idx} overlaps on layers: {overlap}."
            )

        positions = sorted(index_by_layer[layer] for layer in matched_layers)
        for idx in range(1, len(positions)):
            if positions[idx] != positions[idx - 1] + 1:
                raise ValueError(
                    f"Cascade calibration group {group_idx} for block '{block_name}' id {block_id} "
                    "is not consecutive in layer order. "
                    f"Group layers: {matched_layers}."
                )

        grouped_stages.append(matched_layers)
        for layer in matched_layers:
            grouped_set.add(layer)

    ungrouped_layers = [layer for layer in available_layers if layer not in grouped_set]

    if manager.calibration_no_data_dependency:
        merged = []
        seen = set()

        for group_layers in grouped_stages:
            for layer in group_layers:
                if layer not in seen:
                    seen.add(layer)
                    merged.append(layer)

        for layer in ungrouped_layers:
            if layer not in seen:
                seen.add(layer)
                merged.append(layer)

        return [merged] if len(merged) > 0 else []

    stages: List[List[str]] = list(grouped_stages)
    for layer in ungrouped_layers:
        stages.append([layer])
    return stages


def _criteria_match_layer(layer_path: str, block_id: int, criteria) -> bool:
    """Match one layer path against iter_filtered-compatible criteria semantics."""
    if criteria is None:
        return True

    if not isinstance(criteria, list):
        criteria = [criteria]

    # OR across top-level criteria.
    for criterion in criteria:
        if _criterion_matches_layer(layer_path, block_id, criterion):
            return True
    return False


def _criterion_matches_layer(layer_path: str, block_id: int, criterion) -> bool:
    if criterion is None:
        return False

    if criterion in ["all", "ALL", "All"]:
        return True

    if isinstance(criterion, int):
        return block_id == criterion

    if isinstance(criterion, str):
        return criterion in layer_path

    if isinstance(criterion, list):
        # AND within nested list.
        for nested in criterion:
            if not _criterion_matches_layer(layer_path, block_id, nested):
                return False
        return True

    return False


def _resolve_block_container_module(model, path_template: str, block_id: int):
    """
    Resolve the block container module from a template like:
      model.layers.{block_index}.{path}
    """
    split = path_template.split(".{block_index}")
    if len(split) == 0:
        raise ValueError(f"Invalid path_template '{path_template}'.")

    block_prefix = split[0].rstrip(".")
    if len(block_prefix) == 0:
        root = model
    else:
        root = get_submodule(model, block_prefix)

    try:
        return root[block_id]
    except Exception as exc:
        raise ValueError(
            f"Unable to resolve block module for template '{path_template}' and block id {block_id}."
        ) from exc


def _clone_module(module):
    """Prefer module.clone() when available, fallback to deepcopy."""
    clone_fn = getattr(module, "clone", None)
    if callable(clone_fn):
        return clone_fn()
    return copy.deepcopy(module)


def _stage_schemes_from_layer_group(
    block_scheme_dict,
    path_template: str,
    block_id: int,
    layer_group: Sequence[str],
):
    schemes = []
    for layer_path in layer_group:
        full_path = path_template.format(block_index=block_id, path=layer_path)
        scheme = block_scheme_dict.get(full_path, None)
        if scheme is not None:
            schemes.append(scheme)
    return schemes


def _localize_targets_to_block(
    calibration_targets,
    path_template: str,
    block_id: int,
    block_model,
):
    """
    Convert full model paths in calibration targets to block-local paths.

    Calibration runs on a single block module, so a scheme path like:
      model.layers.3.self_attn.q_proj
    must become:
      self_attn.q_proj
    """
    block_prefix = path_template.format(block_index=block_id, path="").rstrip(".")

    class _LocalCalibrationScheme:
        """Minimal scheme adapter expected by calibration backbone."""

        def __init__(self, model, path, calibration_data):
            self.model = model
            self.path = path
            self.calibration_data = calibration_data

        def get_compression_module(self):
            return get_submodule(self.model, self.path)

    localized = []
    for scheme, summary_names in calibration_targets:
        local_path = scheme.path
        if block_prefix and local_path.startswith(block_prefix + "."):
            local_path = local_path[len(block_prefix) + 1 :]

        local_scheme = _LocalCalibrationScheme(
            model=block_model,
            path=local_path,
            calibration_data=scheme.calibration_data,
        )
        localized.append((local_scheme, summary_names))

    return localized


def _collect_preprocessing_outputs(
    preprocessing_module,
    calibration_data,
    max_batches: Optional[int],
    device: torch.device,
    offload_to_cpu: bool,
):
    cached_inputs = []

    with torch.no_grad():
        for batch_id, batch in enumerate(calibration_data):
            if max_batches is not None and batch_id >= max_batches:
                break

            model_args, model_kwargs, label = normalize_calibration_batch(batch, batch_id=batch_id)
            model_args = tuple(move_to_device(model_args, device))
            model_kwargs = dict(move_to_device(model_kwargs, device))

            out = _call_preprocessing_module(
                preprocessing_module=preprocessing_module,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
            out = _detach_and_optionally_offload(out, offload_to_cpu)
            label = _detach_and_optionally_offload(label, offload_to_cpu)

            cached_inputs.append(
                {
                    "__ts_args__": (out,),
                    "__ts_kwargs__": {},
                    "__ts_label__": label,
                }
            )

    if len(cached_inputs) == 0:
        raise ValueError("Calibration data must contain at least one batch.")

    return cached_inputs


def _collect_loader_inputs(
    calibration_data,
    max_batches: Optional[int],
    device: torch.device,
    offload_to_cpu: bool,
):
    cached_inputs = []
    cpu = torch.device("cpu")

    for batch_id, batch in enumerate(calibration_data):
        if max_batches is not None and batch_id >= max_batches:
            break

        model_args, model_kwargs, label = normalize_calibration_batch(batch, batch_id=batch_id)
        model_args = tuple(move_to_device(model_args, device))
        model_kwargs = dict(move_to_device(model_kwargs, device))

        if offload_to_cpu:
            model_args = tuple(move_to_device(model_args, cpu))
            model_kwargs = dict(move_to_device(model_kwargs, cpu))
            label = move_to_device(label, cpu)

        cached_inputs.append(
            {
                "__ts_args__": model_args,
                "__ts_kwargs__": model_kwargs,
                "__ts_label__": label,
            }
        )

    if len(cached_inputs) == 0:
        raise ValueError("Calibration data must contain at least one batch.")

    return cached_inputs


def _run_block_on_cached_inputs(
    block_module,
    cached_inputs,
    device: torch.device,
    offload_to_cpu: bool,
):
    next_cached_inputs = []

    with torch.no_grad():
        for batch in cached_inputs:
            model_args = tuple(move_to_device(batch.get("__ts_args__", tuple()), device))
            model_kwargs = dict(move_to_device(batch.get("__ts_kwargs__", {}), device))
            label = batch.get("__ts_label__", None)

            out = block_module(*model_args, **model_kwargs)
            out = _detach_and_optionally_offload(out, offload_to_cpu)
            label = _detach_and_optionally_offload(label, offload_to_cpu)

            next_cached_inputs.append(
                {
                    "__ts_args__": (out,),
                    "__ts_kwargs__": {},
                    "__ts_label__": label,
                }
            )

    return next_cached_inputs


def _attach_position_embeddings_if_configured(
    cached_inputs,
    block_indexing,
    model,
    device: torch.device,
    offload_to_cpu: bool,
):
    """
    Attach per-batch position_embeddings to cached inputs when indexing declares
    a rotary/position embedding producer for this block family.

    Indexing fields:
      - position_embeddings_source: module path generating position embeddings
      - position_embeddings_targets: subblock names requiring position embeddings
      - position_ids_ndim: 2 (default) or 3 for models requiring 3D RoPE ids
    """
    source_path = block_indexing.get("position_embeddings_source", None)
    target_subblocks = block_indexing.get("position_embeddings_targets", [])

    if source_path is None or len(target_subblocks) == 0:
        return cached_inputs

    position_ids_ndim = int(block_indexing.get("position_ids_ndim", 2))
    if position_ids_ndim not in [2, 3]:
        raise ValueError(
            "position_ids_ndim must be 2 or 3 when position_embeddings_source is set. "
            f"Got {position_ids_ndim}."
        )

    rotary_module = get_submodule(model, source_path)
    cpu = torch.device("cpu")
    enriched = []

    with torch.no_grad():
        for batch in cached_inputs:
            model_args = tuple(move_to_device(batch.get("__ts_args__", tuple()), device))
            model_kwargs = dict(move_to_device(batch.get("__ts_kwargs__", {}), device))
            label = batch.get("__ts_label__", None)

            if len(model_args) == 0 or not isinstance(model_args[0], torch.Tensor):
                enriched.append(batch)
                continue

            hidden_states = model_args[0]
            if hidden_states.ndim < 2:
                enriched.append(batch)
                continue

            batch_size = int(hidden_states.shape[0])
            seq_len = int(hidden_states.shape[1])
            cache_position = torch.arange(seq_len, device=device, dtype=torch.long)

            if position_ids_ndim == 2:
                position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)

            position_embeddings = rotary_module(hidden_states, position_ids)

            if offload_to_cpu:
                model_args = tuple(move_to_device(model_args, cpu))
                model_kwargs = dict(move_to_device(model_kwargs, cpu))
                label = move_to_device(label, cpu)
                position_embeddings = move_to_device(position_embeddings, cpu)
                position_ids = move_to_device(position_ids, cpu)

            model_kwargs = dict(model_kwargs)
            model_kwargs["position_embeddings"] = position_embeddings
            model_kwargs["position_ids"] = position_ids

            enriched.append(
                {
                    "__ts_args__": model_args,
                    "__ts_kwargs__": model_kwargs,
                    "__ts_label__": label,
                }
            )

    return enriched


def _call_preprocessing_module(preprocessing_module, model_args, model_kwargs):
    """
    Execute preprocessing module with a tolerant call strategy.

    Some preprocessing modules (e.g. torch.nn.Embedding) accept only positional
    input and will fail on model-style kwargs like input_ids=....
    """
    try:
        return preprocessing_module(*model_args, **model_kwargs)
    except TypeError as original_exc:
        accepted_kwargs = None

        # Try filtering kwargs based on forward signature first.
        forward_fn = getattr(preprocessing_module, "forward", None)
        if callable(forward_fn):
            try:
                signature = inspect.signature(forward_fn)
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in signature.parameters.values()
                )
                if has_var_keyword:
                    accepted_kwargs = dict(model_kwargs)
                else:
                    accepted_kwargs = {}
                    for name, param in signature.parameters.items():
                        if name == "self":
                            continue
                        if param.kind in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        ) and name in model_kwargs:
                            accepted_kwargs[name] = model_kwargs[name]
            except (TypeError, ValueError):
                accepted_kwargs = None

        if accepted_kwargs is not None and accepted_kwargs != model_kwargs:
            try:
                return preprocessing_module(*model_args, **accepted_kwargs)
            except TypeError:
                pass

        # If args are empty, try passing common model input kwargs as positional.
        if len(model_args) == 0 and len(model_kwargs) > 0:
            preferred_positional_keys = (
                "input_ids",
                "pixel_values",
                "input_values",
                "inputs_embeds",
            )

            for key in preferred_positional_keys:
                if key in model_kwargs:
                    try:
                        return preprocessing_module(model_kwargs[key])
                    except TypeError:
                        pass

            if accepted_kwargs is not None and len(accepted_kwargs) == 1:
                only_value = next(iter(accepted_kwargs.values()))
                try:
                    return preprocessing_module(only_value)
                except TypeError:
                    pass

            if len(model_kwargs) == 1:
                only_value = next(iter(model_kwargs.values()))
                try:
                    return preprocessing_module(only_value)
                except TypeError:
                    pass

        raise original_exc


def _detach_and_optionally_offload(value, offload_to_cpu: bool):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if offload_to_cpu:
            value = value.to(torch.device("cpu"))
        return value

    if isinstance(value, tuple):
        return tuple(_detach_and_optionally_offload(v, offload_to_cpu) for v in value)

    if isinstance(value, list):
        return [_detach_and_optionally_offload(v, offload_to_cpu) for v in value]

    if isinstance(value, dict):
        return {k: _detach_and_optionally_offload(v, offload_to_cpu) for k, v in value.items()}

    return value


__all__ = ["apply_cascade"]
