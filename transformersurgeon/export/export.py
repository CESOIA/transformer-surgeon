import warnings
from typing import Any

import torch.nn as nn

from .registry import EXPORT_ROUTINES
from .config import BackendExportConfig
from ..blocks import TransformerDecoder
from ..utils import convert_for_export


def _is_tsurgeon_decoder(module: nn.Module) -> bool:
    return isinstance(module, TransformerDecoder)


def _resolve_model_components_for_export(
    model_or_graph: Any,
    *,
    convert_options: dict[str, Any] | None = None,
    verbose: bool = False,
) -> tuple[nn.Module, nn.Module, nn.Module, Any | None]:
    """Resolve (embedding, decoder, final_layer, model_config) for export."""
    convert_options = convert_options or {"use_sdpa": False}

    if isinstance(model_or_graph, dict):
        embedding = model_or_graph["embedding"]
        decoder = model_or_graph["decoder"]
        final_layer = model_or_graph["final_layer"]
        config = model_or_graph.get("config")
        return embedding, decoder, final_layer, config

    if isinstance(model_or_graph, (tuple, list)) and len(model_or_graph) == 3:
        embedding, decoder, final_layer = model_or_graph
        return embedding, decoder, final_layer, None

    if hasattr(model_or_graph, "get_input_embeddings") and hasattr(model_or_graph, "lm_head"):
        embedding = model_or_graph.get_input_embeddings()
        final_layer = model_or_graph.lm_head

        if _is_tsurgeon_decoder(model_or_graph):
            decoder = model_or_graph
        else:
            converted = convert_for_export(
                model_or_graph,
                options=convert_options,
                verbose=verbose,
            )
            decoder = converted.get("text")

        if decoder is None:
            raise ValueError(
                "Could not resolve decoder graph model from input model. "
                "Expected convert_for_export(...)[\"text\"] to be available."
            )

        return embedding, decoder, final_layer, getattr(model_or_graph, "config", None)

    raise TypeError(
        "Unsupported model input. Pass either: full model, "
        "dict {embedding, decoder, final_layer}, or "
        "tuple (embedding, decoder, final_layer)."
    )

def export_to_backend(
    model_or_graph: Any,
    config: BackendExportConfig | None = None,
) -> Any:
    """Backend-agnostic export wrapper that delegates to registered backend exporters."""
    if config is None:
        raise ValueError("config must be provided for backend export.")

    # Conversion is backend-agnostic, so normalize to export-ready components here.
    embedding, decoder, final_layer, model_config = _resolve_model_components_for_export(
        model_or_graph,
        convert_options=config.convert_options,
        verbose=config.verbose,
    )
    normalized_model_or_graph = {
        "embedding": embedding,
        "decoder": decoder,
        "final_layer": final_layer,
        "config": model_config,
    }

    export_routine = EXPORT_ROUTINES[config.backend]["export"]
    return export_routine(normalized_model_or_graph, config=config)

# Deprecated
def export_to_executorch(*args, **kwargs):
    """Deprecated alias for export_to_backend."""
    warnings.warn(
        "export_to_executorch is deprecated. Use export_to_backend instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return export_to_backend(*args, **kwargs)

__all__ = [
    "export_to_backend",
    "export_to_executorch",
]