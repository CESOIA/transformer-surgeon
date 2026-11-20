"""Device helpers: find where a model lives, move nested values there, allocate zeros of any dtype."""

from collections.abc import Mapping

import torch


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


def device_zeros(shape, dtype: torch.dtype | None = None, device="cpu") -> torch.Tensor:
    """``torch.zeros`` for any dtype (``None``: the default dtype): one-byte float dtypes are zeroed
    as bytes, since torch cannot fill some of them (``float4_e2m1fn_x2`` on CUDA)."""
    if dtype is not None and dtype.is_floating_point and dtype.itemsize == 1:
        return torch.zeros(shape, dtype=torch.uint8, device=device).view(dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


__all__ = ["infer_model_device", "move_to_device", "device_zeros"]
