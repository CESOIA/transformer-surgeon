"""KV-cache update op for the ``io_inplace`` cache implementation.

``torch.ops.tsurgeon.kv_cache_update(cache, update, write_index)`` returns
``cache`` with ``update`` written along the sequence axis (dim 2) starting at
``write_index``. It is functional in eager PyTorch (the input is not mutated),
which keeps ``torch.export`` graphs pure, while its ONNX translation
(``export/onnx``) is the opset-24 ``TensorScatter`` op: TensorRT maps that to
its in-place KV-cache update layer and aliases the output to the input buffer,
so no per-step cache copy is made.

Layout is BHSD: cache ``(1, kv_heads, max_cache_len, head_dim)``, update
``(1, kv_heads, seq_len, head_dim)``, write_index ``(1,)`` int64.
"""

import torch


@torch.library.custom_op("tsurgeon::kv_cache_update", mutates_args=())
def kv_cache_update(cache: torch.Tensor, update: torch.Tensor, write_index: torch.Tensor) -> torch.Tensor:
    positions = write_index + torch.arange(update.shape[2], device=cache.device)
    return cache.index_copy(2, positions, update)


@kv_cache_update.register_fake
def _(cache, update, write_index):
    return torch.empty_like(cache)


__all__ = ["kv_cache_update"]
