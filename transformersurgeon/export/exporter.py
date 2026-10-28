"""Generic exporter: the steps every backend shares, and the backend registry.

``export_to_backend(model, config)`` picks the exporter registered for
``config.backend`` and runs :meth:`Exporter.export`:

1. complete the ``convert_options`` with the backend defaults of
   ``export/operations.py`` and reject unsupported choices;
2. resolve the export components (input embedding, converted decoder stack,
   output head), converting the Hugging Face model with ``convert_for_export``
   when needed;
3. move the components to CPU (tracing device) and to the requested dtype;
4. call the backend's :meth:`Exporter.lower`, which traces, quantises, lowers
   and writes the artifact, and returns the result.

A backend is added by subclassing :class:`Exporter` in ``export/backends/`` and
decorating it with :func:`register_exporter`.
"""

import os
from typing import Any

from ..blocks import TransformerDecoder
from ..utils import get_submodule
from .config import BackendExportConfig
from .convert import convert_for_export
from .operations import check_components, check_config_choices, resolve_convert_options
from .wrapper import _normalize_component_devices

EXPORTERS: dict[str, type] = {}


def register_exporter(cls):
    """Class decorator: register an :class:`Exporter` subclass under its ``backend`` name."""
    EXPORTERS[cls.backend] = cls
    return cls


def get_exporter(backend):
    """Return an instance of the exporter registered for ``backend``."""
    from . import backends  # noqa: F401  (registers the built-in exporters)

    if backend not in EXPORTERS:
        raise ValueError(f"Unknown export backend {backend!r}; available: {sorted(EXPORTERS)}.")
    return EXPORTERS[backend]()


def _decoder_component(model):
    """Name and model dictionary of the component converted into the decoder stack."""
    for name, indexing in getattr(model, "indexing", {}).items():
        if indexing.get("structure") == "transformer_decoder":
            return name, indexing
    return None, {}


def resolve_components(model_or_graph: Any, *, convert_options=None, verbose=False):
    """Return ``(embedding, decoder, final_layer, model_config)`` for export.

    Accepts a dict ``{embedding, decoder, final_layer[, config]}``, a 3-tuple, or
    a compressed Hugging Face causal language model. For a model, embedding and
    output head are taken from the decoder component's dictionary
    (``preprocessing``, ``final_layer``) so their compression is kept, falling
    back to ``get_input_embeddings()``/``lm_head``.
    """
    if isinstance(model_or_graph, dict):
        return (model_or_graph["embedding"], model_or_graph["decoder"], model_or_graph["final_layer"],
                model_or_graph.get("config"))
    if isinstance(model_or_graph, (tuple, list)) and len(model_or_graph) == 3:
        return (*model_or_graph, None)
    if hasattr(model_or_graph, "get_input_embeddings") and hasattr(model_or_graph, "lm_head"):
        name, indexing = _decoder_component(model_or_graph)
        embedding = get_submodule(model_or_graph, indexing["preprocessing"]) if indexing.get("preprocessing") else None
        final_layer = get_submodule(model_or_graph, indexing["final_layer"]) if indexing.get("final_layer") else None
        embedding = embedding if embedding is not None else model_or_graph.get_input_embeddings()
        final_layer = final_layer if final_layer is not None else model_or_graph.lm_head
        if isinstance(model_or_graph, TransformerDecoder):
            decoder = model_or_graph
        else:
            decoder = convert_for_export(model_or_graph, options=convert_options, verbose=verbose).get(name)
        if decoder is None:
            raise ValueError("Could not convert a decoder from the model: no component of its dictionary "
                             "declares structure='transformer_decoder'.")
        return embedding, decoder, final_layer, getattr(model_or_graph, "config", None)
    raise TypeError("Unsupported model input. Pass either: full model, dict {embedding, decoder, final_layer}, "
                    "or tuple (embedding, decoder, final_layer).")


class Exporter:
    """Base class of the backend exporters (template method :meth:`export`).

    Subclasses set ``backend`` and ``config_class`` and implement :meth:`lower`.
    """

    backend: str = None
    config_class: type = BackendExportConfig

    def export(self, model_or_graph: Any, config: BackendExportConfig):
        if not isinstance(config, self.config_class):
            raise TypeError(f"The {self.backend} exporter needs a {self.config_class.__name__}, "
                            f"got {type(config).__name__}.")
        self.check_available()
        check_config_choices(self.backend, config)

        options = dict(config.convert_options or {})
        if "max_cache_len" not in options and isinstance(config.max_seq_len, int) and config.max_seq_len > 0:
            options["max_cache_len"] = config.max_seq_len
        options = resolve_convert_options(self.backend, options)

        embedding, decoder, final_layer, model_config = resolve_components(
            model_or_graph, convert_options=options, verbose=config.verbose)
        check_components(self.backend, decoder)
        embedding, decoder, final_layer = _normalize_component_devices(embedding, decoder, final_layer)
        if config.float_type is not None:
            embedding, decoder, final_layer = (m.to(config.float_type) for m in (embedding, decoder, final_layer))

        os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
        components = {"embedding": embedding, "decoder": decoder, "final_layer": final_layer, "config": model_config}
        return self.lower(components, config)

    def check_available(self):
        """Raise if the backend's toolchain is missing (default: always available)."""

    def lower(self, components: dict, config: BackendExportConfig):
        """Trace, quantise, lower and write ``components``; return the export result."""
        raise NotImplementedError


def export_to_backend(model_or_graph: Any, config: BackendExportConfig | None = None):
    """Export a compressed model (or already-converted components) with the backend named by ``config.backend``."""
    if config is None:
        raise ValueError("config must be provided for backend export.")
    return get_exporter(config.backend).export(model_or_graph, config)


__all__ = ["Exporter", "EXPORTERS", "register_exporter", "get_exporter", "resolve_components", "export_to_backend"]
