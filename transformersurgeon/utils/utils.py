from collections.abc import Mapping

import torch


def flatten_index_paths(path_list):
    """Normalize indexing path definitions to a flat list of dotted paths."""
    if isinstance(path_list, dict):
        flattened_paths = []
        for parent_path, child_paths in path_list.items():
            if not isinstance(child_paths, (list, tuple)):
                raise TypeError(
                    "Grouped path_list values must be a list/tuple of sublayers. "
                    f"Found {type(child_paths).__name__} for key '{parent_path}'."
                )

            if len(child_paths) == 0:
                flattened_paths.append(parent_path)
                continue

            for child_path in child_paths:
                child_path = str(child_path)
                if parent_path:
                    flattened_paths.append(f"{parent_path}.{child_path}")
                else:
                    flattened_paths.append(child_path)

        return flattened_paths

    if isinstance(path_list, (list, tuple)):
        return list(path_list)

    raise TypeError(
        "Indexing 'path_list' must be a list/tuple or grouped dict, "
        f"got {type(path_list).__name__}."
    )


def get_submodule(module, submodule_path):
    """
    Returns the submodule of a given module based on the dot-separated path.

    Args:
        module: The parent module from which to retrieve the submodule.

    Returns:
        The submodule located at the specified path.
    """
    split_path = submodule_path.split('.')
    # Traverse the module iteratively to find the submodule
    tmp_module = module
    for path_piece in split_path:
        tmp_module = getattr(tmp_module, path_piece, None)

        if tmp_module is None:
            raise ValueError(f"Module at path '{submodule_path}' not found in module {module.__class__.__name__}.")

    return tmp_module


def infer_model_device(model) -> torch.device:
    """
    Infer the most likely device for a model.

    Resolution order:
    1) model.device (if available)
    2) first parameter device
    3) first buffer device
    4) cpu fallback
    """
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        return torch.device("cpu")


def move_to_device(value, target_device: torch.device):
    """
    Recursively move tensors and tensor-like objects to a target device.

    Supports tensors, mappings, tuples, lists, and objects exposing `.to(...)`.
    Non-tensor values are returned unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.to(target_device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, target_device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, target_device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, target_device) for item in value]
    if hasattr(value, "to"):
        try:
            return value.to(target_device)
        except TypeError:
            pass
    return value


__all__ = ["get_submodule", "infer_model_device", "move_to_device", "flatten_index_paths"]