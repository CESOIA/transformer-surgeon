"""Runtime manifest written next to every exported artifact.

The manifest describes the graph contract (cache atom, layouts, per-layer cache
shapes and input/output names) so runners never re-derive geometry.
"""

import json
import os
from typing import Any

import torch
import torch.nn as nn

from ..blocks import KV_CACHES
from ..utils.devices import device_zeros


MANIFEST_FORMAT = "tsurgeon-llm/1"


def manifest_path_for(artifact_path: str) -> str:
    """``<artifact stem>.manifest.json`` next to an exported artifact."""
    return os.path.splitext(artifact_path)[0] + ".manifest.json"


def cache_io_name(kind: str, layer: int, output: bool = False) -> str:
    """Graph I/O name of one layer's cache: ``past_<kind>_<i>`` in, ``present_<kind>_<i>`` out
    (``kind`` is ``key``, ``value``, their block scales ``key_scale``/``value_scale``, or ``kv``
    for a paged pool)."""
    return f"{'present' if output else 'past'}_{kind}_{layer}"


def cache_scales(decoder: nn.Module) -> bool:
    """Whether each key/value cache of ``decoder`` is a ``(rows, scales)`` pair (block scales)."""
    return any(getattr(block.attn, "cache_scale_dtype", None) is not None for block in getattr(decoder, "blocks", ()))


def graph_io_names(num_layers: int, paged: bool = False, scales: bool = False) -> tuple[list[str], list[str]]:
    """``(input_names, output_names)`` of a graph-I/O cache wrapper, in flattened argument order
    (``scales``: each key/value cache is a ``(rows, scales)`` pair)."""
    if paged:
        groups = (("kv",),)
    else:
        groups = tuple((kind, f"{kind}_scale") if scales else (kind,) for kind in ("key", "value"))
    flat = [(kind, i) for group in groups for i in range(num_layers) for kind in group]
    inputs = ["input_ids", "pos_id"] + [cache_io_name(kind, i) for kind, i in flat]
    outputs = ["logits"] + [cache_io_name(kind, i, output=True) for kind, i in flat]
    return inputs, outputs


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def build_llm_manifest(wrapper: nn.Module, model_config: Any | None, *, backend: str,
                       artifact: str, **extra: Any) -> dict[str, Any]:
    """Describe an exported LLM's runtime contract, for any backend.

    Runners use it instead of re-deriving geometry: the cache implementation,
    layout and dtype, and one entry per layer with the key/value cache shapes
    (which differ per layer after structured pruning), dtypes and graph I/O names;
    a precision with block scales adds ``key_scale``/``value_scale`` entries, one more
    tensor per cache. Backend-specific fields go in ``extra``.
    """
    decoder = wrapper.decoder
    cache_impl = getattr(decoder, "cache_impl", "mutable")
    kv_cache = KV_CACHES[cache_impl]
    kinds = ("kv",) if cache_impl == "io_paged" else ("key", "value")
    if cache_scales(decoder):
        kinds += ("key_scale", "value_scale")

    def entry(attn, kind, i):
        dtype = attn.cache_scale_dtype if kind.endswith("_scale") else attn.key_cache.dtype
        return {"input": cache_io_name(kind, i), "output": cache_io_name(kind, i, output=True),
                "shape": list(attn.cache_shape(kind)), "dtype": _dtype_name(dtype)}

    caches = [{"layer": i, **{k: entry(block.attn, k, i) for k in kinds}} for i, block in enumerate(decoder.blocks)]
    eos = getattr(model_config, "eos_token_id", None) if model_config is not None else None
    return {
        "format": MANIFEST_FORMAT,
        "backend": backend,
        "artifact": os.path.basename(artifact),
        "cache_impl": cache_impl,
        "cache_io": cache_impl != "mutable",
        "cache_inplace": cache_impl in ("io_inplace", "io_paged"),
        # present_* are only the new rows; the caller writes them at pos_id (write_cache_rows).
        "cache_rows": kv_cache.returns_rows,
        "cache_layout": kv_cache.manifest_layout,
        # The allocated cache buffer, not config.dtype (None for configs without one).
        "cache_dtype": _dtype_name(decoder.blocks[0].attn.key_cache.dtype),
        # "full", "int8"/"int4" (uint8 rows of codes + a 2-byte scale), "fp8" (float8_e4m3fn rows) or
        # "nvfp4" (float4_e2m1fn_x2 rows + float8_e4m3fn block scales), see blocks/attention/kv_precision.
        "kv_precision": getattr(decoder.blocks[0].attn, "kv_precision", "full"),
        "max_cache_len": int(decoder.max_cache_len),
        "num_layers": len(decoder.blocks),
        "add_batch_dim": bool(getattr(wrapper, "add_batch_dim", False)),
        "attn_impl": getattr(decoder.blocks[0].attn, "attn_impl", None) if len(decoder.blocks) else None,
        # RoPE variant baked into the graph's cos/sin tables (indexing 'rope_type').
        "rope_type": getattr(getattr(decoder, "config", None), "rope_type", "default"),
        "vocab_size": int(getattr(model_config, "vocab_size", 0) or 0) if model_config is not None else 0,
        "eos_token_id": eos if isinstance(eos, list) else ([eos] if eos is not None else []),
        "caches": caches,
        **extra,
    }


def load_llm_manifest(artifact_path: str) -> dict[str, Any]:
    """Load the manifest written next to an exported artifact (or given directly)."""
    path = artifact_path if artifact_path.endswith(".manifest.json") else manifest_path_for(artifact_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No export manifest at '{path}'. Re-export with this version of transformer-surgeon "
            "(every backend now writes <artifact>.manifest.json)."
        )
    with open(path) as f:
        return json.load(f)


def zero_caches_from_manifest(
    manifest: dict[str, Any], device: torch.device | str = "cpu"
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-layer zero (key, value) caches with the exported shapes and dtypes, ``(rows, scales)``
    pairs with block scales (io_paged: (kv pools, []) )."""
    def zeros(spec):
        return device_zeros(spec["shape"], getattr(torch, spec.get("dtype", manifest["cache_dtype"])), device)

    def cache(layer, kind):
        return (zeros(layer[kind]), zeros(layer[f"{kind}_scale"])) if f"{kind}_scale" in layer else zeros(layer[kind])

    if manifest.get("cache_layout") == "PAGED":
        return [zeros(layer["kv"]) for layer in manifest["caches"]], []
    layers = manifest["caches"]
    return [cache(layer, "key") for layer in layers], [cache(layer, "value") for layer in layers]


def write_cache_rows(cache: torch.Tensor, rows: torch.Tensor, pos_id: torch.Tensor) -> None:
    """``io_rows`` contract: write the rows a call returned (``present_*``, ``(1, kv_heads, seq, width)``)
    into the caller's BHSD cache at ``pos_id``, in place. Works inside a CUDA graph (``pos_id`` may
    live on the device); one-byte rows (uint8, FP8, FP4 pairs) are copied as bytes. A
    ``(rows, scales)`` cache takes the ``(rows, scales)`` a call returned."""
    if isinstance(cache, (tuple, list)):
        for part, part_rows in zip(cache, rows):
            write_cache_rows(part, part_rows, pos_id)
        return
    positions = pos_id.view(-1)[:1] + torch.arange(rows.shape[2], device=rows.device)
    if cache.element_size() == 1:
        cache, rows = cache.view(torch.uint8), rows.view(torch.uint8)
    cache.index_copy_(2, positions, rows)


__all__ = ["MANIFEST_FORMAT", "manifest_path_for", "cache_io_name", "cache_scales", "graph_io_names",
           "build_llm_manifest", "load_llm_manifest", "zero_caches_from_manifest", "write_cache_rows"]
