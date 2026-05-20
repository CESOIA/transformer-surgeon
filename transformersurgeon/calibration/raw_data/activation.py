"""Activation raw data collector."""

from __future__ import annotations

from typing import Callable

import torch

from .base import RawDataCollector


class ActivationCollector(RawDataCollector):
    name = "activation"

    def __init__(self, *, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu

    def build_forward_hook(self, *, emit_raw: Callable[[str, torch.Tensor], None]):
        def _hook(module, inputs, _output):
            if len(inputs) == 0:
                raise ValueError("Cannot collect calibration data from a module without input activations.")

            activation = inputs[0].detach()
            in_features = getattr(module, "in_features", activation.size(-1))
            if activation.size(-1) != in_features:
                raise ValueError(
                    f"Calibration activation last dimension is {activation.size(-1)}, expected {in_features}."
                )

            if activation.dim() == 1:
                activation = activation.reshape(1, -1)
            else:
                activation = activation.reshape(-1, activation.size(-1))

            if self.offload_to_cpu:
                activation = activation.cpu()

            emit_raw(self.name, activation.float())

        return _hook
