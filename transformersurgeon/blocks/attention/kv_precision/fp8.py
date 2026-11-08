"""Keys and values stored as FP8 (E4M3) numbers.

``torch.ops.tsurgeon.kv_fp8_quantize(x)`` rounds float rows to ``float8_e4m3fn``
(nearest even, saturating at +-448) and ``torch.ops.tsurgeon.kv_fp8_dequantize(rows)``
widens them back to float32. Their ONNX translations (``export/onnx``) are
``QuantizeLinear``/``DequantizeLinear`` with a unit scale: TensorRT only creates
FP8 tensors with a quantize layer and only reads them with a dequantize layer
(an ONNX ``Cast`` to FP8 is refused).
"""

import torch

from .base import KVPrecision

FP8_MAX = 448.0  # largest finite float8_e4m3fn


@torch.library.custom_op("tsurgeon::kv_fp8_quantize", mutates_args=())
def kv_fp8_quantize(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.float(), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


@kv_fp8_quantize.register_fake
def _(x):
    return torch.empty_like(x, dtype=torch.float8_e4m3fn)


@torch.library.custom_op("tsurgeon::kv_fp8_dequantize", mutates_args=())
def kv_fp8_dequantize(rows: torch.Tensor) -> torch.Tensor:
    return rows.float()


@kv_fp8_dequantize.register_fake
def _(rows):
    return torch.empty_like(rows, dtype=torch.float32)


class FP8Precision(KVPrecision):
    """One ``float8_e4m3fn`` element per value with a unit scale, as FP8 KV caches are used
    when no calibrated scale is given: 3 mantissa bits (relative error at most 1/16),
    values beyond +-448 saturate, values below 2**-6 lose precision (subnormals).

    Needs ``cache_impl="io_rows"``: FP8 rows can be produced and read, but not moved
    by a scatter or select inside the graph (TensorRT refuses it; torch has no FP8
    ``index_copy``), and ``io_rows`` leaves the write to the caller.
    """

    name = "fp8"
    bits = 8
    requires_cache = "io_rows"

    def storage_dtype(self, dtype):
        return torch.float8_e4m3fn

    def row_width(self, head_dim):
        return head_dim

    def encode(self, x):
        return kv_fp8_quantize(x)

    def decode(self, rows, dtype):
        return kv_fp8_dequantize(rows).to(dtype)


__all__ = ["FP8_MAX", "FP8Precision", "kv_fp8_quantize", "kv_fp8_dequantize"]
