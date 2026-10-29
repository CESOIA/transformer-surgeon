"""XNNPACK exporter (ExecuTorch, CPU)."""

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.export import export as torch_export

from ..exporter import register_exporter
from ..operations import config_default
from ..pt2e import calibrate_pt2e_observers, inject_scales_into_pt2e_observers
from ..pt2e_linear_quantizer import LinearOnlyQuantizer
from ..quant_info import _PRECISION_TO_QRANGE, extract_layer_quant_info
from ..config import BackendExportConfig
from ..wrapper import resolve_components_and_wrapper
from .executorch_base import ExecuTorchExporter


@dataclass
class XNNPACKExportConfig(BackendExportConfig):
    """XNNPACK export options.

    quant_path: how quantised linear layers are lowered (see the
        ``linear_quantization`` entry of ``export/operations.py``):
        ``"pt2e"`` (default) or ``"torchao"``.
    calibration_token_ids: token-id sequences used to calibrate activation
        observers (``None``: a few built-in sentences).
    """

    backend: str = "xnnpack"
    quant_path: str = field(default_factory=lambda: config_default("xnnpack", "quant_path"))
    calibration_token_ids: Any = None


def build_quantizer_from_layer_info(layer_info: dict[str, dict[str, Any]]):
    """Build a per-module linear-only quantizer driven by compression metadata,
    with ExecuTorch's XNNPACK quantization configs.

    Each layer in layer_info gets its own qconfig; layers absent from
    layer_info are not quantised (they remain float). Only linear nodes are
    annotated (see linear_quantizer.py for why the stock XNNPACKQuantizer,
    which also annotates attention's add/mul/cat, is not used).

    Layers with stored activation calibration data use static quantization
    (is_dynamic=False); others use dynamic weight-only (is_dynamic=True).
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import get_symmetric_quantization_config


    quantizer = LinearOnlyQuantizer()
    for layer_name, info in layer_info.items():
        qrange = _PRECISION_TO_QRANGE.get(info["precision"])
        if qrange is None:
            warnings.warn(
                f"Unsupported compression precision {info['precision']} for "
                f"layer '{layer_name}'; skipping — layer will remain float.",
                stacklevel=2,
            )
            continue

        if info.get("group_size") is not None:
            raise NotImplementedError(
                f"XNNPACK PT2E quantizer: '{layer_name}' is per-group quantized; the PT2E "
                "path only expresses per-tensor/per-channel weights.")
        is_dynamic = info["act_scale"] is None  # True → weight-only (fp activations); False → static (calibrated activations)
        qconfig = get_symmetric_quantization_config(
            is_per_channel=info["per_channel"],
            is_dynamic=is_dynamic,
            **qrange,
        )
        quantizer.set_module_name(layer_name, qconfig)

    return quantizer


def _torchao_quantize_linears(wrapper, layer_info: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Swap each quantized linear's (already dequantized) weight for an
    IntxUnpackedToInt8Tensor carrying tsurgeon's exact scales. Returns
    {layer_name: granularity} for the layers converted."""
    import torch.nn as nn
    from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
        IntxUnpackedToInt8Tensor,
        IntxUnpackedToInt8TensorActivationQuantization,
    )

    modules = dict(wrapper.named_modules())
    done = {}
    for name, info in layer_info.items():
        if info.get("act_scale") is not None:
            raise NotImplementedError(
                f"XNNPACK torchao path: '{name}' has static activation calibration; only dynamic "
                "per-token int8 activations are supported on this path.")
        if not info.get("hard") or info.get("scale") is None:
            raise NotImplementedError(f"XNNPACK torchao path: '{name}' is not hard-quantized.")
        if not info.get("act_dynamic"):
            warnings.warn(
                f"XNNPACK torchao path: '{name}' is weight-only (no dynamic int8 activations); "
                "XNNPACK has no fp32 x intN weight-only GEMM, so this layer may not be delegated.",
                stacklevel=2)
        bits = int(info["precision"])
        if not 2 <= bits <= 8:
            raise ValueError(f"XNNPACK torchao path: unsupported precision {bits} for '{name}'.")
        mod = modules[name]
        w = mod.weight.detach()
        n, k = w.shape
        scale = info["scale"].detach().to(w.dtype)
        g = info.get("group_size")
        if g is not None:
            scale = scale.reshape(n, k // g)
            block = (1, g)
            gran = f"group{g}"
        elif info.get("per_channel"):
            scale = scale.reshape(n, 1)
            block = (1, k)
            gran = "per_channel"
        else:
            scale = scale.reshape(1, 1).expand(n, 1).contiguous()
            block = (1, k)
            gran = "per_tensor"
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
        qdata = torch.clamp(
            torch.round(w.float() / scale.float().repeat_interleave(block[1], dim=1)), qmin, qmax
        ).to(torch.int8)
        qw = IntxUnpackedToInt8Tensor(
            qdata=qdata,
            scale=scale,
            zero_point=torch.zeros_like(scale, dtype=torch.int8),
            target_dtype=getattr(torch, f"int{bits}"),
            block_size=block,
            dtype=w.dtype,
            activation_quantization=(IntxUnpackedToInt8TensorActivationQuantization.INT8_ASYM_PER_TOKEN
                                     if info.get("act_dynamic") else None),
        )
        mod.weight = nn.Parameter(qw, requires_grad=False)
        done[name] = gran
    return done


@register_exporter
class XNNPACKExporter(ExecuTorchExporter):
    """Traces the wrapper, applies the quantisation read from the layers, delegates to XNNPACK."""

    backend = "xnnpack"
    config_class = XNNPACKExportConfig

    def lower(self, components, config):
        wrapper, model_config, example_inputs = resolve_components_and_wrapper(components)
        layer_info = extract_layer_quant_info(wrapper)
        if config.quant_path == "torchao":
            return self._lower_torchao(wrapper, model_config, example_inputs, layer_info, config)

        exported = torch_export(wrapper, example_inputs)
        export_for_edge, exported_for_mismatch = exported, None
        if layer_info:
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            prepared = prepare_pt2e(exported.module(), build_quantizer_from_layer_info(layer_info))
            # Real text, not random tokens: random tokens leave output observers with
            # degenerate ranges (gate/up outputs mostly negative -> positive values clipped).
            calibrate_pt2e_observers(prepared, model_config, config, example_inputs=example_inputs)
            # Hard-quantised layers use the exact scales of the compression step.
            inject_scales_into_pt2e_observers(prepared, layer_info)
            export_for_edge = exported_for_mismatch = torch_export(convert_pt2e(prepared), example_inputs)

        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        edge = to_edge_transform_and_lower(
            export_for_edge,
            compile_config=EdgeCompileConfig(_check_ir_validity=config.check_ir_validity),
            partitioner=[XnnpackPartitioner()],
        )
        self.write_program(edge, config, alloc_graph_input=False)
        return self.finalize(config, precision="mixed" if layer_info else "full", wrapper=wrapper,
                             example_inputs=example_inputs, model_config=model_config,
                             exported_for_mismatch=exported_for_mismatch)

    def _lower_torchao(self, wrapper, model_config, example_inputs, layer_info, config):
        """torchao 8daXw linears, dynamically-quantised partitioner first."""
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackDynamicallyQuantizedPartitioner,
            XnnpackPartitioner,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        converted = _torchao_quantize_linears(wrapper, layer_info) if layer_info else {}
        with torch.no_grad():
            exported = torch_export(wrapper, example_inputs)
        edge = to_edge_transform_and_lower(
            exported,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()],
        )
        self.write_program(edge, config, alloc_graph_input=False)
        result = self.finalize(config, precision="mixed" if layer_info else "full", wrapper=wrapper,
                               example_inputs=example_inputs, model_config=model_config,
                               exported_for_mismatch=None, run_weight_mismatch_check=False)
        result.quantized_layers = converted
        return result


__all__ = ["XNNPACKExportConfig", "XNNPACKExporter", "build_quantizer_from_layer_info"]
