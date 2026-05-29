"""Weight gradient raw data collector."""

from __future__ import annotations

from typing import Optional

import torch

from .base import RawDataCollector


class WeightGradCollector(RawDataCollector):
    name = "weight_grad"
    # Gradient data is only available after loss.backward().
    requires_backward = True
    # A differentiable calibration loss is required and provided by the manager.
    requires_loss = True
    # Labels are optional and fully defined by the user-provided loss function.
    requires_labels = False

    def __init__(self, *, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu

    def collect_after_backward(self, module) -> Optional[torch.Tensor]:
        # Pull dLoss/dW from the compressed module's weight parameter.
        weight_grad = getattr(module.weight, "grad", None)
        if weight_grad is None:
            return None
        # Snapshot gradient values for this batch to avoid aliasing reused grad buffers.
        grad = weight_grad.detach().clone()
        if self.offload_to_cpu:
            grad = grad.cpu()
        return grad.float()
