"""
pruning_dims.py

Dependency-free helpers for turning a structured-pruning ratio into an effective
(kept) output dimension. This is the single source of truth for the
ratio -> kept-neuron conversion, shared by the compression algorithms
(``compression/structured_pruning_methods``) and the converted-graph blocks
(``blocks/mlp.py``, ``blocks/mha.py``) so a pruned model and its exported graph
agree on shapes.

Lives in ``blocks`` (the lowest layer) because ``compression`` already imports
``blocks``; keeping the helper here avoids a blocks -> compression import cycle.
"""

from typing import Union


def effective_num_pruned(
    dim: int,
    ratio: float,
    granularity: Union[str, int] = "layer",
) -> int:
    """Number of output neurons removed for a layer of ``dim`` output rows.

    - ``granularity == "layer"`` (or ``None``): remove ``int(ratio * dim)`` rows.
    - ``granularity == g`` (positive int): remove ``int(ratio * g)`` rows within
      each consecutive chunk of ``g`` (uniform per chunk, e.g. per attention
      head). ``g`` must divide ``dim``.
    """
    if ratio is None or ratio <= 0.0:
        return 0
    if ratio >= 1.0:
        return dim

    if granularity in ("layer", None):
        return int(ratio * dim)

    g = int(granularity)
    if g <= 0:
        raise ValueError(f"granularity must be a positive int or 'layer', got {granularity!r}.")
    if dim % g != 0:
        raise ValueError(f"granularity {g} must evenly divide the output dimension {dim}.")
    per_chunk = int(ratio * g)
    num_chunks = dim // g
    return per_chunk * num_chunks


def effective_out_features(
    dim: int,
    ratio: float,
    granularity: Union[str, int] = "layer",
) -> int:
    """Kept output-neuron count after structured pruning (``dim`` minus pruned)."""
    return dim - effective_num_pruned(dim, ratio, granularity)


def pruning_config_dims(layer_config: dict, dim: int) -> int:
    """Effective ``dim`` after applying a layer's structured_pruning config.

    Reads ``ratio`` and ``granularity`` from ``layer_config['structured_pruning']``
    (missing keys default to no pruning) and returns the kept output dimension.
    Safe to call with any per-layer compression-config dict.
    """
    sp = (layer_config or {}).get("structured_pruning", {}) or {}
    ratio = sp.get("ratio", 0.0)
    granularity = sp.get("granularity", "layer")
    return effective_out_features(dim, ratio, granularity)


__all__ = [
    "effective_num_pruned",
    "effective_out_features",
    "pruning_config_dims",
]
