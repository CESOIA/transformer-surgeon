"""Weight/bias parameter raw data collectors.

Unlike activation streams, these are not activation-dependent -- they capture
the module's own parameters. They still go through a forward hook (like every
other collector) so the 'weight'/'bias' summaries can be requested through the
same generic calibration pipeline as any other summary.
"""

from __future__ import annotations

from typing import Callable

import torch

from .base import RawDataCollector


class WeightCollector(RawDataCollector):
    name = "weight"

    def __init__(self, *, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu

    def build_forward_hook(self, *, emit_raw: Callable[[str, torch.Tensor], None]):
        def _hook(module, inputs, output):
            weight = module.weight.detach().clone()
            if self.offload_to_cpu:
                weight = weight.cpu()
            emit_raw(self.name, weight.float())

        return _hook


class BiasCollector(RawDataCollector):
    name = "bias"

    def __init__(self, *, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu

    def build_forward_hook(self, *, emit_raw: Callable[[str, torch.Tensor], None]):
        def _hook(module, inputs, output):
            bias = getattr(module, "bias", None)
            if bias is None:
                raise ValueError(
                    f"Cannot collect 'bias' calibration summary: module {type(module).__name__} has no bias."
                )
            bias = bias.detach().clone()
            if self.offload_to_cpu:
                bias = bias.cpu()
            emit_raw(self.name, bias.float())

        return _hook
