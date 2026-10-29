"""Traceable wrapper around the export components and its example inputs.

Every backend traces the same :class:`LLMWrapper`: input embedding, converted
decoder stack and output head, with a fixed signature that depends only on the
KV-cache atom of the decoder (internal state, graph input/output, or pages).
"""

from typing import Any

import torch
import torch.nn as nn


def _graph_io_cache(decoder):
    kv_cache = getattr(decoder, "kv_cache", None)
    return kv_cache is not None and kv_cache.graph_io


def build_zero_caches(decoder: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Zero KV caches (one per block) for a decoder whose cache is a graph input/output.

    Shapes come from each block's attention module, so per-layer geometry after
    pruning is respected. Returns ``([], [])`` for an internal cache; for pages,
    one pool per block in the first list and an empty second list.
    """
    if not _graph_io_cache(decoder):
        return [], []
    if decoder.cache_impl == "io_paged":
        return [torch.zeros(b.attn.cache_shape("kv"), dtype=b.attn.dtype) for b in decoder.blocks], []
    key_caches = [torch.zeros(b.attn.cache_shape("key"), dtype=b.attn.dtype) for b in decoder.blocks]
    value_caches = [torch.zeros(b.attn.cache_shape("value"), dtype=b.attn.dtype) for b in decoder.blocks]
    return key_caches, value_caches


class LLMWrapper(nn.Module):
    """Thin wrapper that presents the three model components under a fixed
    forward signature that torch.export can trace with a single token input.

    The signature depends on the decoder's ``cache_impl``:
      - ``mutable``: ``forward(input_ids, pos_id)`` -> ``logits`` (KV cache is
        internal module state; unchanged legacy contract).
      - ``io_*``: ``forward(input_ids, pos_id, key_caches, value_caches)`` ->
        ``(logits, new_key_caches, new_value_caches)`` (KV cache is graph I/O).
      - ``io_paged``: ``forward(input_ids, pos_id, kv_pools)`` ->
        ``(logits, new_kv_pools)`` (one paged K+V pool per layer).
    """

    def __init__(self, embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.final_layer = final_layer
        self.cache_impl = getattr(decoder, "cache_impl", "mutable")
        self.graph_io_cache = _graph_io_cache(decoder)
        # With add_batch_dim, input_ids is (1, seq) and every hidden state keeps
        # the batch dim; only the final last-token slice changes.
        self.add_batch_dim = getattr(decoder, "add_batch_dim", False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        pos_id_tensor: torch.LongTensor,
        key_caches: list[torch.Tensor] | None = None,
        value_caches: list[torch.Tensor] | None = None,
    ):
        if not self.graph_io_cache:
            hidden = self.decoder(self.embedding(input_ids), pos_id=pos_id_tensor)
            return self._head(hidden)

        if self.cache_impl == "io_paged":
            hidden, new_pools, _ = self.decoder(self.embedding(input_ids), pos_id=pos_id_tensor, key_caches=key_caches)
            return self._head(hidden), new_pools

        hidden, new_key_caches, new_value_caches = self.decoder(
            self.embedding(input_ids),
            pos_id=pos_id_tensor,
            key_caches=key_caches,
            value_caches=value_caches,
        )
        return self._head(hidden), new_key_caches, new_value_caches

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Take the last token's hidden state and project it to logits.

        The slice deliberately keeps a leading dim (``hidden[-1:, :]`` rather
        than ``hidden[-1, :]``) so ``final_layer`` always sees a rank-2 input,
        and the rank is dropped again afterwards. The output contract is
        unchanged -- ``(vocab,)`` unbatched, ``(1, vocab)`` with add_batch_dim --
        but a rank-1 linear input breaks QNN lowering: ExecuTorch's
        ConvertLinearToConv2d reshapes a rank-1 input to rank 2 and then applies
        a 4-element permutation to it, which raises "input.dim() = 2 is not equal
        to len(dims) = 4". Rank-2 is also the shape QNN's FullyConnected builder
        handles most robustly, which is why LinearCompressed flattens to rank 2
        for its own inputs.
        """
        if self.add_batch_dim:
            return self.final_layer(hidden[:, -1, :])
        return self.final_layer(hidden[-1:, :])[0]


def _normalize_component_devices(
    embedding: nn.Module, decoder: nn.Module, final_layer: nn.Module, *, device: str = "cpu",
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Move export components onto a common device before tracing.

    torch.export requires every tensor a traced module touches to share one
    device, and build_example_inputs() always builds CPU tensors, so CPU is
    the correct common device for every backend. Backends that need the
    final artifact elsewhere (currently only TensorRT, via config.device)
    re-place it themselves after tracing/compiling.
    """
    return embedding.to(device), decoder.to(device), final_layer.to(device)


def build_wrapper(components) -> nn.Module:
    """``LLMWrapper`` from a dict ``{embedding, decoder, final_layer}`` or a 3-tuple."""
    if isinstance(components, dict):
        components = (components["embedding"], components["decoder"], components["final_layer"])
    if not (isinstance(components, (tuple, list)) and len(components) == 3):
        raise TypeError("build_wrapper expects dict {embedding, decoder, final_layer} "
                        "or tuple/list (embedding, decoder, final_layer).")
    return LLMWrapper(*components)


def build_example_inputs(model_config: Any | None, *, add_batch_dim: bool = False) -> tuple[Any, ...]:
    """Single-token example inputs ``(input_ids, pos_id)`` for tracing and calibration."""
    vocab_size = int(getattr(model_config, "vocab_size", 100)) if model_config is not None else 100
    input_ids = torch.randint(0, vocab_size, (1, 1) if add_batch_dim else (1,), dtype=torch.long)
    return (input_ids, torch.tensor([1], dtype=torch.long))


def resolve_components_and_wrapper(components: dict) -> tuple[nn.Module, Any | None, tuple[Any, ...]]:
    """``(wrapper, model_config, example_inputs)`` for already-resolved export components.

    For a graph-I/O cache, zero caches are appended to the example inputs so
    ``torch.export`` traces the full contract.
    """
    wrapper = build_wrapper(components).eval()
    model_config = components.get("config")
    example_inputs = build_example_inputs(model_config, add_batch_dim=wrapper.add_batch_dim)
    if wrapper.graph_io_cache:
        key_caches, value_caches = build_zero_caches(wrapper.decoder)
        if wrapper.cache_impl == "io_paged":
            example_inputs = (*example_inputs, key_caches)
        else:
            example_inputs = (*example_inputs, key_caches, value_caches)
    return wrapper, model_config, example_inputs


__all__ = [
    "build_zero_caches", "LLMWrapper", "build_wrapper", "build_example_inputs", "resolve_components_and_wrapper",
]
