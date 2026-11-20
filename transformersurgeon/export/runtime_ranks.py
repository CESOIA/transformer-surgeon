"""Runtime rank inputs of an exported graph.

LRD layers with a ``runtime_rank_group`` (``blocks/layers/runtime_rank.py``) can run at any rank up to the
one they store. An export exposes each selected group as one graph input, ``rank_<group>``: a 1-D int32
tensor whose *length* is the active rank (its values are never read). The traced graph slices the
factors with that length (``Shape -> Slice`` in ONNX), so a TensorRT engine built with an optimization
profile over the length runs every rank of the range with one set of weights.

``resolve_runtime_ranks`` decides which groups become inputs and their ``(min, opt, max)`` range; the
manifest records the result under ``runtime_ranks`` and runtimes read it from there.
"""

from __future__ import annotations

import re
from typing import Mapping

import torch
import torch.nn as nn

from ..blocks.layers import runtime_rank_groups

RANK_INPUT_DTYPE = torch.int32


def rank_input_name(group: str) -> str:
    """Graph input name of a runtime rank group."""
    return "rank_" + re.sub(r"\W", "_", group)


def resolve_runtime_ranks(module: nn.Module, requested: Mapping[str, tuple | None] | None = None) -> dict[str, dict]:
    """``{group: {"input", "min", "opt", "max", "layers"}}`` of the runtime rank inputs to export.

    ``requested`` maps a group to its ``(min, opt, max)`` range (``None``: ``(1, stored, stored)``);
    ``None`` selects every group found in ``module`` with the default range, ``{}`` none (every layer
    then runs at its stored rank). The layers of one group must store the same rank.
    """
    groups = runtime_rank_groups(module)
    if requested is None:
        requested = {group: None for group in groups}
    unknown = sorted(set(requested) - set(groups))
    if unknown:
        raise ValueError(f"Runtime rank groups {unknown} not found; the model has {sorted(groups)}.")
    resolved = {}
    for group, bounds in requested.items():
        layers = groups[group]
        stored = {layer.rank for _, layer in layers}
        if len(stored) != 1:
            raise ValueError(f"Runtime rank group {group!r} mixes stored ranks {sorted(stored)}: an exported "
                             "rank input needs one stored rank per group (use one group per rank).")
        stored = stored.pop()
        lo, opt, hi = bounds if bounds is not None else (1, stored, stored)
        if not 1 <= lo <= opt <= hi <= stored:
            raise ValueError(f"Runtime rank group {group!r}: need 1 <= min <= opt <= max <= {stored} "
                             f"(stored rank), got {(lo, opt, hi)}.")
        resolved[group] = {"input": rank_input_name(group), "min": int(lo), "opt": int(opt), "max": int(hi),
                           "layers": [name for name, _ in layers]}
    return resolved


def runtime_rank_example_inputs(ranks: Mapping[str, dict]) -> tuple[torch.Tensor, ...]:
    """One example tensor per rank input, of a traceable length inside its range."""
    examples = []
    for spec in ranks.values():
        length = spec["opt"] if spec["opt"] >= 2 else min(2, spec["max"])
        examples.append(torch.zeros(length, dtype=RANK_INPUT_DTYPE))
    return tuple(examples)


def runtime_rank_dynamic_shapes(ranks: Mapping[str, dict]) -> tuple:
    """``torch.export`` dynamic shapes of the rank inputs (a static input for a fixed range)."""
    shapes = []
    for spec in ranks.values():
        if spec["min"] == spec["max"]:
            shapes.append(None)
        else:
            shapes.append({0: torch.export.Dim(spec["input"], min=spec["min"], max=spec["max"])})
    return tuple(shapes)


__all__ = ["RANK_INPUT_DTYPE", "rank_input_name", "resolve_runtime_ranks", "runtime_rank_example_inputs",
           "runtime_rank_dynamic_shapes"]
