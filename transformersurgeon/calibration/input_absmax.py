"""Per-input-channel maximum absolute activation of selected modules.

A standalone calibration pass (it needs no compression scheme): it runs the
model over the calibration data and records, for every listed module, the
largest ``|x|`` seen on each input channel. Used by SmoothQuant.
"""

from typing import Iterable, Optional

import torch
import torch.nn as nn

from ..utils import infer_model_device, move_to_device, normalize_batch


@torch.no_grad()
def collect_input_absmax(model: nn.Module, modules: Iterable[nn.Module], calibration_data,
                         max_batches: Optional[int] = None) -> dict[nn.Module, torch.Tensor]:
    """Per-input-channel max |x| of each module's input over the calibration data (fp32)."""
    stats: dict[nn.Module, torch.Tensor] = {}
    handles = []

    def make_hook(mod):
        def hook(_, inputs):
            x = inputs[0].detach().float().abs().reshape(-1, inputs[0].shape[-1]).amax(0)
            stats[mod] = x if mod not in stats else torch.maximum(stats[mod], x)
        return hook

    for m in set(modules):
        handles.append(m.register_forward_pre_hook(make_hook(m)))
    device = infer_model_device(model)
    try:
        for batch_id, batch in enumerate(calibration_data):
            if max_batches is not None and batch_id >= max_batches:
                break
            args, kwargs, _ = normalize_batch(batch, batch_id=batch_id)
            model(*move_to_device(args, device), **move_to_device(kwargs, device))
    finally:
        for h in handles:
            h.remove()
    return {m: v.cpu() for m, v in stats.items()}


__all__ = ["collect_input_absmax"]
