# `blocks/` — the export-ready transformer, in atoms

`blocks/` is a simplified transformer that `export/convert.py` builds from an **already compressed** Hugging
Face model. It never decides what to compress: shapes, ranks and biases are read from the source layers.

It is split into **atoms**, one component per file, grouped in sub-folders:

```
layers/      LinearCompressed, EmbeddingCompressed, Conv2d/3dCompressed, LowRankMixin, VCONBlock, AtomicSum, LayerScale
norm/        rmsnorm, rmsnorm_prescaled, rmsnorm_upcast, layernorm
mlp/         plain, gated, activations
positions/   rope, frequency tables, rope_pruning
attention/   base (MHABase), causal (MHACausal), bidirectional (MHAEncoder)
  kernels/   manual, sdpa, custom_sdpa, paged
  kv_cache/  mutable, io_scatter, io_concat, io_inplace, io_paged
stacks/      decoder, encoder, config
```

## The block catalog

`blocks/catalog.py::BLOCK_CATALOG` lists every atom: where it is implemented, what it does, which operation it
provides (atoms with the same `implements` are interchangeable) and who selects it — the **model** dictionary
(the model needs that operation) or the **export** dictionary (the device prefers that implementation).
`build_block(family, name, ...)` instantiates an atom from the catalog; no code hard-codes a choice.

--8<-- "docs/generated/block_catalog.md"

## Special roles

Exporters that must treat an atom specially read its `EXPORT_ROLE` class attribute (`"attention"` on
`MHACausal`, `"decoder_stack"` on `TransformerDecoder`) instead of matching class names.

API: [blocks](../api/blocks.md).
