"""Keys and values stored as NVFP4: FP4 (E2M1) numbers with one FP8 (E4M3) scale per 16 elements.

A cache is two tensors: the rows, ``float4_e2m1fn_x2`` (two codes per byte, element ``2i``
in the low nibble, bit 3 of a code is the sign), and their block scales,
``float8_e4m3fn``, ``head_dim / 16`` per row.
``torch.ops.tsurgeon.kv_nvfp4_quantize(x)`` returns ``(rows, scales)``: each block of 16
elements gets the scale ``max|x| / 6`` rounded to E4M3 (nearest even, saturating at 448)
and each element ``x / scale`` rounded to E2M1 (nearest even, saturating at +-6).
``torch.ops.tsurgeon.kv_nvfp4_dequantize(rows, scales)`` widens them back to float32.
The per-tensor (second-level) scale of NVFP4 is 1. Their ONNX translations
(``export/backends/onnx``) are a ``QuantizeLinear`` of the block scales to FLOAT8E4M3FN followed by
a blocked ``QuantizeLinear`` to FLOAT4E2M1, and the two ``DequantizeLinear`` back: the
only form in which TensorRT creates and reads FP4 tensors.
"""

import torch

from .base import KVPrecision
from .fp8 import FP8_MAX

NVFP4_BLOCK = 16   # elements per FP8 block scale
E2M1_MAX = 6.0     # largest FP4 (E2M1) magnitude
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # magnitude of each 3-bit code
# Rounding thresholds between consecutive magnitudes, and whether a tie rounds up: ties go to the even code.
_E2M1_STEPS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))


def _e2m1_codes(y):
    """E2M1 codes (``uint8``, sign in bit 3) of ``y`` rounded to the nearest even, saturating at +-6.
    The sign bit follows ``y``'s sign, -0.0 included, as TensorRT's quantize sets it."""
    magnitude = y.abs()
    codes = torch.zeros(y.shape, dtype=torch.uint8, device=y.device)
    for threshold, tie_up in _E2M1_STEPS:
        codes += (magnitude >= threshold) if tie_up else (magnitude > threshold)
    return codes | (torch.signbit(y).to(torch.uint8) << 3)


@torch.library.custom_op("tsurgeon::kv_nvfp4_quantize", mutates_args=())
def kv_nvfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = x.float().unflatten(-1, (-1, NVFP4_BLOCK))
    scales = torch.clamp(blocks.abs().amax(dim=-1) / E2M1_MAX, max=FP8_MAX).to(torch.float8_e4m3fn)
    step = scales.float()
    step = torch.where(step > 0, step, torch.ones_like(step))   # an all-zero block keeps zero codes
    codes = _e2m1_codes(blocks / step.unsqueeze(-1)).flatten(-2)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2), scales


@kv_nvfp4_quantize.register_fake
def _(x):
    return (x.new_empty(*x.shape[:-1], x.shape[-1] // 2, dtype=torch.float4_e2m1fn_x2),
            x.new_empty(*x.shape[:-1], x.shape[-1] // NVFP4_BLOCK, dtype=torch.float8_e4m3fn))


@torch.library.custom_op("tsurgeon::kv_nvfp4_dequantize", mutates_args=())
def kv_nvfp4_dequantize(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed = rows.view(torch.uint8)
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2).long()
    table = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=rows.device)
    values = table[codes].unflatten(-1, (-1, NVFP4_BLOCK))
    return (values * scales.float().unsqueeze(-1)).flatten(-2)


@kv_nvfp4_dequantize.register_fake
def _(rows, scales):
    return rows.new_empty(*rows.shape[:-1], 2 * rows.shape[-1], dtype=torch.float32)


class NVFP4Precision(KVPrecision):
    """NVFP4 as TensorRT stores it: 4-bit E2M1 numbers (magnitudes 0, 0.5, 1, 1.5, 2, 3,
    4, 6) and one E4M3 scale per block of 16 elements, i.e. 4.5 bits per element. The
    scale follows the block's largest value, so the rounding error is relative to the
    block: at most 1/5 of a value above 1/6 of the block's largest, coarser below. With
    the unit per-tensor scale, a block whose largest value is below 6 * 2**-10 is stored
    as zero and one below 6 * 2**-6 has a subnormal scale.

    A cache is the pair ``(rows, scales)``: the rows are ``float4_e2m1fn_x2`` (two codes per
    byte), the scales a second ``float8_e4m3fn`` tensor (``scale_dtype``). Needs a head dim
    that is a multiple of 16 and ``cache_impl="io_rows"``: FP4 and FP8 rows can be produced
    and read, but not moved by a scatter or select inside the graph.
    """

    name = "nvfp4"
    bits = 4
    requires_cache = "io_rows"
    scale_dtype = torch.float8_e4m3fn

    def storage_dtype(self, dtype):
        if not hasattr(torch, "float4_e2m1fn_x2"):
            raise RuntimeError("kv_precision='nvfp4' needs a PyTorch with torch.float4_e2m1fn_x2.")
        return torch.float4_e2m1fn_x2

    def row_width(self, head_dim):
        if head_dim % NVFP4_BLOCK:
            raise ValueError(f"kv_precision='nvfp4' stores one scale per {NVFP4_BLOCK} elements and needs a head "
                             f"dim that is a multiple of {NVFP4_BLOCK}, got {head_dim}.")
        return head_dim // 2

    def scale_width(self, head_dim):
        self.row_width(head_dim)   # checks the head dim
        return head_dim // NVFP4_BLOCK

    def encode(self, x):
        return kv_nvfp4_quantize(x)

    def decode(self, rows, dtype):
        return kv_nvfp4_dequantize(*rows).to(dtype)


__all__ = ["NVFP4_BLOCK", "E2M1_MAX", "E2M1_VALUES", "NVFP4Precision", "kv_nvfp4_quantize", "kv_nvfp4_dequantize"]
