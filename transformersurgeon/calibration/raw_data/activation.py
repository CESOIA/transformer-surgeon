"""Activation raw data collector."""

from __future__ import annotations

from typing import Callable

import torch

from .base import RawDataCollector


class ActivationCollector(RawDataCollector):
    name = "activation"

    def __init__(self, *, offload_to_cpu: bool = False, raw_name: str = "activation"):
        self.offload_to_cpu = offload_to_cpu
        self.name = raw_name

    def build_forward_hook(self, *, emit_raw: Callable[[str, torch.Tensor], None]):
        def _hook(module, inputs, _output):
            # Activation summaries consume the module input tensor (pre-linear activations).
            if len(inputs) == 0:
                raise ValueError("Cannot collect calibration data from a module without input activations.")

            activation = inputs[0].detach()
            # For linear-like modules, last dim must match in_features.
            in_features = getattr(module, "in_features", activation.size(-1))
            if activation.size(-1) != in_features:
                raise ValueError(
                    f"Calibration activation last dimension is {activation.size(-1)}, expected {in_features}."
                )

            # Flatten batch/time axes to a 2D [tokens, features] matrix.
            if activation.dim() == 1:
                activation = activation.reshape(1, -1)
            else:
                activation = activation.reshape(-1, activation.size(-1))

            # Optional CPU offload reduces GPU peak memory during long calibration runs.
            if self.offload_to_cpu:
                activation = activation.cpu()

            # Emit float activations so summaries run in a stable common dtype.
            emit_raw(self.name, activation.float())

        return _hook


class ShiftedActivationCollector(ActivationCollector):
    name = "activation_shifted"
    # This stream is collected from the shifted model.
    uses_shifted_model = True

    def __init__(self, *, offload_to_cpu: bool = False):
        super().__init__(offload_to_cpu=offload_to_cpu, raw_name=self.name)


class OutputActivationCollector(RawDataCollector):
    """Collects the OUTPUT tensor of a module (post-linear activations).

    Mirrors ActivationCollector but captures the module's return value rather
    than inputs[0].  Used by OutputActivationRangeSummary to calibrate the
    output-side quantization observer required by XNNPACK's fused INT8 kernel.
    """

    name = "output_activation"

    def __init__(self, *, offload_to_cpu: bool = False):
        self.offload_to_cpu = offload_to_cpu

    def build_forward_hook(self, *, emit_raw: Callable[[str, torch.Tensor], None]):
        def _hook(module, inputs, output):
            act = output.detach()
            out_features = getattr(module, "out_features", act.size(-1))
            if act.size(-1) != out_features:
                raise ValueError(
                    f"Output activation last dimension is {act.size(-1)}, expected {out_features}."
                )
            if act.dim() == 1:
                act = act.reshape(1, -1)
            else:
                act = act.reshape(-1, act.size(-1))
            if self.offload_to_cpu:
                act = act.cpu()
            emit_raw(self.name, act.float())

        return _hook
