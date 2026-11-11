"""KV-cache precision atoms: how one cached key/value row is stored. One atom per file."""
from .base import KVPrecision
from .integer import IntegerKVPrecision, SCALE_BYTES
from .full import FullPrecision
from .int8 import Int8Precision
from .int4 import Int4Precision
from .fp8 import FP8_MAX, FP8Precision, kv_fp8_quantize, kv_fp8_dequantize
from .nvfp4 import NVFP4_BLOCK, NVFP4Precision, kv_nvfp4_quantize, kv_nvfp4_dequantize

KV_PRECISIONS = {
    precision.name: precision
    for precision in (FullPrecision(), Int8Precision(), Int4Precision(), FP8Precision(), NVFP4Precision())
}

__all__ = [
    "KVPrecision", "IntegerKVPrecision", "SCALE_BYTES", "FullPrecision", "Int8Precision", "Int4Precision",
    "FP8_MAX", "FP8Precision", "kv_fp8_quantize", "kv_fp8_dequantize",
    "NVFP4_BLOCK", "NVFP4Precision", "kv_nvfp4_quantize", "kv_nvfp4_dequantize", "KV_PRECISIONS",
]
