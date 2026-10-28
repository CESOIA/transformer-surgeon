"""ARCHIVED: unused pieces of the old export/common.py (2026-10-28 refactor).

- SUPPORTED_QUANT_CONFIGS: precision labels referenced only by a re-export, never read.
- to_device: duplicate of transformersurgeon.utils.move_to_device, no callers.
- ExporterConfig: abstract layer whose only field, dynamic_shapes, was always
  ignored (the exporters trace a static single-token contract).
"""
from abc import ABC
from dataclasses import dataclass
from typing import Any

import torch

from transformersurgeon.export.config import BackendExportConfig


SUPPORTED_QUANT_CONFIGS: dict[str, dict] = {
    "full":  {"weight_bits": None, "activation_bits": None, "description": "No quantization (FP32)"},
    "w4":    {"weight_bits": 4,    "activation_bits": None, "description": "4-bit weight-only"},
    "w8":    {"weight_bits": 8,    "activation_bits": None, "description": "8-bit weight-only"},
    "a8w8":  {"weight_bits": 8,    "activation_bits": 8,    "description": "8-bit activations + 8-bit weights"},
    "a8w4":  {"weight_bits": 4,    "activation_bits": 8,    "description": "8-bit activations + 4-bit weights"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass


def to_device(x: Any, device: torch.device | str) -> Any:
    """Move a tensor, or a (nested) list/tuple of tensors, to ``device``.

    Needed for the io_* cache modes, whose KV caches are nested list args.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(e, device) for e in x)
    return x


@dataclass
class ExporterConfig(BackendExportConfig, ABC):
    """Abstract base config shared by all torch.export-based backend exporters."""

    dynamic_shapes: dict[str, Any] | None = None