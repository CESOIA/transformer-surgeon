"""KV-cache precision atoms: how one cached key/value row is stored. One atom per file."""
from .base import KVPrecision, IntegerKVPrecision, SCALE_BYTES
from .full import FullPrecision
from .int8 import Int8Precision
from .int4 import Int4Precision

KV_PRECISIONS = {
    precision.name: precision
    for precision in (FullPrecision(), Int8Precision(), Int4Precision())
}

__all__ = [
    "KVPrecision", "IntegerKVPrecision", "SCALE_BYTES", "FullPrecision", "Int8Precision", "Int4Precision",
    "KV_PRECISIONS",
]
