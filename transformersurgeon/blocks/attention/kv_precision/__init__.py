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


def kv_precision_pair(precision) -> tuple[str, str]:
    """``(key precision, value precision)`` of a ``kv_precision`` choice: one name of ``KV_PRECISIONS`` for both
    caches, or a ``(key, value)`` pair of names (a list is accepted, e.g. read from JSON)."""
    pair = (precision, precision) if isinstance(precision, str) else precision
    if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(p not in KV_PRECISIONS for p in pair):
        raise ValueError(f"Unsupported kv_precision {precision!r}; expected one of {tuple(KV_PRECISIONS)} "
                         "or a (key, value) pair of them.")
    return tuple(pair)


def canonical_kv_precision(precision):
    """A ``kv_precision`` choice in its canonical form: the name when keys and values share it, else the
    ``(key, value)`` tuple."""
    key, value = kv_precision_pair(precision)
    return key if key == value else (key, value)

__all__ = [
    "KVPrecision", "IntegerKVPrecision", "SCALE_BYTES", "FullPrecision", "Int8Precision", "Int4Precision",
    "FP8_MAX", "FP8Precision", "kv_fp8_quantize", "kv_fp8_dequantize",
    "NVFP4_BLOCK", "NVFP4Precision", "kv_nvfp4_quantize", "kv_nvfp4_dequantize", "KV_PRECISIONS",
    "kv_precision_pair", "canonical_kv_precision",
]
