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


class ParamCollector(RawDataCollector):
    """Emits a float32 copy of the module parameter named ``name`` on every forward."""

    def __init__(self, *, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu

    def build_forward_hook(self, *, emit_raw: Callable[[str, torch.Tensor], None]):
        def _hook(module, inputs, output):
            param = getattr(module, self.name, None)
            if param is None:
                raise ValueError(
                    f"Cannot collect '{self.name}' calibration summary: module {type(module).__name__} "
                    f"has no {self.name}."
                )
            param = param.detach().clone()
            if self.offload_to_cpu:
                param = param.cpu()
            emit_raw(self.name, param.float())

        return _hook


class WeightCollector(ParamCollector):
    name = "weight"


class BiasCollector(ParamCollector):
    name = "bias"
