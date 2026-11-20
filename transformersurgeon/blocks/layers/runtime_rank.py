"""Runtime-selectable rank of low-rank (LRD) layers.

An LRD layer stores ``weight`` ``[out, rank]`` (U) and ``linear_V.weight`` ``[rank, in]`` (V). When its
components are sorted by decreasing importance (the compression step records this as ``rank_ordered``),
the first ``r`` of them are that method's rank-``r`` solution, so the same stored factors can run at any
smaller rank: ``y = U[:, :r] (V[:r] x)``, with the cost of a rank-``r`` layer.

Layers that share one knob carry the same ``runtime_rank_group`` label (set by the compression step).
The active rank of each group holds for the forward passes run inside :func:`active_ranks`:

    with active_ranks({"mlp": 256}):
        logits = model(pixel_values)

A rank is an ``int`` (eager use) or a 1-D tensor whose *length* is the rank: that is how an exported
graph receives it (a graph input with a dynamic dimension, so the traced graph slices the factors with a
symbolic size; its values are never read). Groups missing from the mapping run at their stored rank, and
an ``int`` larger than a layer's stored rank is clamped to it.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Mapping

import torch
import torch.nn as nn

_ACTIVE_RANKS: contextvars.ContextVar = contextvars.ContextVar("tsurgeon_active_ranks", default=None)


@contextmanager
def active_ranks(ranks: Mapping[str, "int | torch.Tensor"]):
    """Run the enclosed forward passes with ``{group: rank}`` (nested calls override per group)."""
    for group, rank in ranks.items():
        if isinstance(rank, torch.Tensor):
            if rank.dim() != 1:
                raise ValueError(f"Runtime rank of group {group!r}: a tensor rank must be 1-D (its length "
                                 f"is the rank), got shape {tuple(rank.shape)}.")
        elif not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError(f"Runtime rank of group {group!r} must be a positive int or a 1-D tensor, got {rank!r}.")
    token = _ACTIVE_RANKS.set({**(_ACTIVE_RANKS.get() or {}), **ranks})
    try:
        yield
    finally:
        _ACTIVE_RANKS.reset(token)


def active_rank(group: str | None):
    """Active rank of ``group`` (an int, or a symbolic size while tracing), or ``None`` if not set."""
    if group is None:
        return None
    ranks = _ACTIVE_RANKS.get()
    if not ranks or group not in ranks:
        return None
    rank = ranks[group]
    return rank.shape[0] if isinstance(rank, torch.Tensor) else rank


def runtime_rank_groups(module: nn.Module) -> dict[str, list[tuple[str, nn.Module]]]:
    """``{group: [(name, layer), ...]}`` of the LRD layers in ``module`` that have a runtime rank group."""
    groups: dict[str, list[tuple[str, nn.Module]]] = {}
    for name, sub in module.named_modules():
        group = getattr(sub, "runtime_rank_group", None)
        if group is not None and isinstance(getattr(sub, "rank", "full"), int):
            groups.setdefault(group, []).append((name, sub))
    return dict(sorted(groups.items()))


__all__ = ["active_ranks", "active_rank", "runtime_rank_groups"]
