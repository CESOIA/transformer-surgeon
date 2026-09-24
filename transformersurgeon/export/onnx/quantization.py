"""
Weight-only Q/DQ for the ONNX exporter.

The ONNX graph is exported with float weights (extract_layer_quant_info
dequantizes hard-quantized layers first). This pass re-expresses every
quantized linear's weight initializer as an integer initializer plus a
DequantizeLinear, which TensorRT fuses into a weight-only GEMM:

  INT8  per-channel  DequantizeLinear(axis=0), scale (out,)
  INT4  block-wise   DequantizeLinear(axis=1, block_size=B), scale (out, in/B)
                     per-channel INT4 repeats each channel scale over B=128
                     blocks (exact, and a kernel-supported block size)

Hard-quantized layers reuse tsurgeon's exact scales; soft-quantized ones
(scale None) get symmetric per-channel max-abs scales. Activations stay float
(activation calibration, if any, is reported and ignored).
"""

import warnings
from typing import Any

import numpy as np

_QRANGE = {8: (-127, 127), 4: (-8, 7)}
# Smallest normal fp16: any positive scale is exact for an all-zero channel or
# block, but anything smaller flushes to 0 once cast to the model dtype, and
# TensorRT rejects non-positive DequantizeLinear scales.
_SCALE_FLOOR = 2.0 ** -14


def _default_int4_block(in_features: int) -> int | None:
    """Largest of 128/64/32 that divides in_features and is *smaller* than it.

    TensorRT 10.16 produces NaNs for a block spanning the whole row, so a
    per-channel layer needs at least two blocks; None if in_features < 64.
    """
    for block in (128, 64, 32):
        if in_features % block == 0 and block < in_features:
            return block
    return None


def _find_initializer(model, name: str):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


def apply_weight_quantization(model, layer_info: dict[str, dict[str, Any]], *, int4_block_size: int | None = None) -> dict[str, dict[str, Any]]:
    """Rewrite quantized linear weights of ``model`` (in place) as Q/DQ.

    Returns {layer_name: {"precision", "granularity"}} for the layers rewritten.
    ``int4_block_size`` (None = per-channel) sets the INT4 DQ block size.
    """
    import ml_dtypes
    from onnx import helper, numpy_helper

    done: dict[str, dict[str, Any]] = {}
    for layer_name, info in layer_info.items():
        precision = int(info["precision"])
        if precision not in _QRANGE:
            warnings.warn(f"ONNX export: unsupported precision {precision} for {layer_name}; kept float.", stacklevel=2)
            continue
        if info.get("act_scale") is not None:
            warnings.warn(f"ONNX export: activation quantization of {layer_name} is not exported "
                          "(weight-only); activations stay float.", stacklevel=2)

        weight_name = f"{layer_name}.weight"
        init = _find_initializer(model, weight_name)
        if init is None:
            warnings.warn(f"ONNX export: weight '{weight_name}' not found as an initializer; kept float.", stacklevel=2)
            continue
        w = numpy_helper.to_array(init).astype(np.float32)  # (out, in)
        out_features, in_features = w.shape
        qmin, qmax = _QRANGE[precision]
        float_dtype = numpy_helper.to_array(init).dtype

        if info.get("scale") is not None:
            scale = info["scale"].detach().float().cpu().numpy().reshape(-1)
            if scale.size == 1:
                scale = np.full(out_features, scale.item(), np.float32)
            scale = np.maximum(scale, _SCALE_FLOOR)
        else:
            scale = np.maximum(np.abs(w).max(axis=1) / qmax, _SCALE_FLOOR)

        if precision == 8:
            q = np.clip(np.rint(w / scale[:, None]), qmin, qmax).astype(np.int8)
            q_init = numpy_helper.from_array(q, weight_name + "_q")
            scale_init = numpy_helper.from_array(scale.astype(float_dtype), weight_name + "_scale")
            dq = helper.make_node("DequantizeLinear", [q_init.name, scale_init.name], [weight_name + "_dq"],
                                  axis=0, name=weight_name + "_DequantizeLinear")
            granularity = "per_channel"
        else:
            if int4_block_size is None:
                # Per-channel INT4, expressed as blocks that all repeat the
                # channel scale: numerically identical, but a block size the
                # INT4 GEMM kernels support. TensorRT 10.16 returns NaNs for a
                # single row-wide block (block_size == in_features).
                block = _default_int4_block(in_features)
                if block is None:
                    warnings.warn(f"ONNX export: {layer_name} (in_features={in_features}) is too narrow "
                                  "for INT4 blocks; kept float.", stacklevel=2)
                    continue
                block_scale = np.repeat(scale.reshape(out_features, 1), in_features // block, axis=1)
                granularity = "per_channel"
            else:
                block = int4_block_size
                if in_features % block:
                    raise ValueError(f"{layer_name}: in_features {in_features} not divisible by INT4 block {block}.")
                # tsurgeon's scales are per output channel; finer blocks recompute
                # max-abs scales per block.
                blocks = w.reshape(out_features, in_features // block, block)
                block_scale = np.maximum(np.abs(blocks).max(axis=2) / qmax, _SCALE_FLOOR)
                granularity = f"block{block}"
            q = np.clip(np.rint(w / np.repeat(block_scale, block, axis=1)), qmin, qmax)
            q_init = numpy_helper.from_array(q.astype(ml_dtypes.int4), weight_name + "_q")
            scale_init = numpy_helper.from_array(block_scale.astype(float_dtype), weight_name + "_scale")
            dq = helper.make_node("DequantizeLinear", [q_init.name, scale_init.name], [weight_name + "_dq"],
                                  axis=1, block_size=block, name=weight_name + "_DequantizeLinear")

        model.graph.initializer.remove(init)
        model.graph.initializer.extend([q_init, scale_init])
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == weight_name:
                    node.input[i] = dq.output[0]
        model.graph.node.insert(0, dq)
        done[layer_name] = {"precision": precision, "granularity": granularity}
    return done


__all__ = ["apply_weight_quantization"]
