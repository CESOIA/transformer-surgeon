# TransformerSurgeon Framework Structure

How the `transformersurgeon` package is laid out and how a model flows through it, from compression to a
device artifact. The binding rules for each folder are in [AGENTS.md](AGENTS.md#architecture).

## Package Layout

```
transformersurgeon/
├── utils/          generic helpers only
│   ├── modules.py          get/set submodule, replace_parameter
│   ├── indexing.py         flatten_index_paths, get_block_config, num_blocks
│   ├── batches.py          normalize_batch, pack_batch
│   ├── devices.py, progress.py
├── models/         one folder per family, two dictionaries each
│   ├── factory.py          build_family(): generates config / model / manager classes
│   └── <m>_c/              indexing_<m>_c.py (model dictionary), family_<m>_c.py (family dictionary)
├── compression/    deciding and applying compression
│   ├── manager.py          CompressionSchemesManager (criteria, calibration, apply/restore, groups)
│   ├── scheme.py           CompressionScheme (one layer; owns its topology: coupled_target_chains)
│   ├── grouping.py         SchemeGroup (shared masks)
│   ├── cascade.py          block-wise (cascade) calibration
│   ├── config.py           init_compressed_config
│   ├── compressible_layers.py  replace_layers_upon_init
│   ├── registry.py, validators.py   options, defaults, validation
│   ├── abstract.py         Compressor base class
│   ├── lrd.py, structured_pruning.py, block_pruning.py, coupled_pruning.py,
│   │   unstructured_pruning.py, quantization.py, smoothquant.py
│   └── *_methods/          algorithm variants (one file each, listed in METHOD_FUNCTIONS)
├── calibration/    collecting statistics
│   ├── backbone.py         run_compression_calibration
│   ├── raw_data/           forward/backward hooks (registry.py)
│   ├── summaries/          running statistics (registry.py)
│   ├── input_absmax.py     per-channel max |x| (SmoothQuant)
│   └── datasets.py         token-window calibration loaders (WikiText-2)
├── blocks/         export-ready transformer, split into selectable atoms
│   ├── catalog.py          BLOCK_CATALOG: family -> atom -> {object, summary, implements, selected_by}
│   ├── layers/             LinearCompressed, EmbeddingCompressed, Conv2d/3dCompressed, LowRankMixin, VCONBlock, AtomicSum, LayerScale
│   ├── norm/               rmsnorm, rmsnorm_prescaled, rmsnorm_upcast, layernorm
│   ├── mlp/                plain, gated, activations
│   ├── positions/          rope, frequency tables, rope_pruning
│   ├── attention/          base (MHABase), causal (MHACausal), bidirectional (MHAEncoder)
│   │   ├── kernels/        manual, sdpa, custom_sdpa, paged
│   │   └── kv_cache/       mutable, io_scatter, io_concat, io_inplace, io_paged
│   └── stacks/             decoder, encoder, config
├── export/         from compressed model to device artifact
│   ├── operations.py       BACKEND_OPERATIONS: block type -> device operation per backend
│   ├── convert.py          convert_for_export: HF model -> blocks/ stacks
│   ├── exporter.py         Exporter, register_exporter, export_to_backend
│   ├── wrapper.py, quant_info.py, pt2e.py, pt2e_linear_quantizer.py, checks.py, manifest.py, result.py
│   └── backends/           executorch_base.py, xnnpack.py, qnn/, onnx/, tensorrt/
└── hf/
    └── hf_export.py        export_to_hf (local directory and Hub, remote code)
```

## End-to-End Data Flow

1. **Model classes.** `models/<m>_c/family_<m>_c.py` calls `build_family`, which generates
   `<Name>ConfigCompress`, `<Name>...Compress` and `<Name>CompressionSchemesManager` from the family dictionary.
   At init the model replaces the layers listed in the model dictionary with compressible ones
   (`replace_layers_upon_init`), and the configuration fills a per-layer `compression_config`.
2. **Configuration.** `manager.set(type, option, value, criteria=...)` writes options on the matching schemes
   (validated by `compression/registry.py`); nothing touches the model yet. `manager.auto_groups()` builds
   shared-mask groups from the dictionary's `pruning.coupled_masks*`.
3. **Calibration (optional).** `manager.set_calibration_data(loader)`; the compressors declare which statistics
   they need; `calibration/` collects them in `"standard"` (one pass) or `"cascade"` (block by block, using the
   dictionary's `calibration_groups`) mode. `manager.smooth_activations(...)` applies SmoothQuant first if needed.
4. **Apply.** `manager.apply(hard=False)` simulates the compression reversibly; `hard=True` makes it permanent
   (rows removed, factors stored, integer weights). Structured pruning cascades into the coupled layers resolved
   by `CompressionScheme.coupled_target_chains`.
5. **Convert.** `convert_for_export(model, options)` builds, for each component of the model dictionary with a
   `structure`, a `blocks/` stack from the atoms the dictionary names and the implementations the options choose
   (filled per backend from `BACKEND_OPERATIONS`). Shapes, ranks, biases and quantisation state are copied from the
   source layers.
6. **Export.** `export_to_backend(model, config)` wraps embedding, decoder and head in `LLMWrapper`, reads the
   per-layer quantisation (`extract_layer_quant_info`), traces, quantises and lowers with the backend exporter, and
   writes the artifact and its manifest.
7. **Or save for Hugging Face.** `export_to_hf(model, repo_id, manager=...)` saves a directory loadable with
   `trust_remote_code`.

## Extending

Recipes (model family, compression method or type, calibration summary, atom, backend) are in
[AGENTS.md](AGENTS.md#extension-recipes). Each addition stays inside one folder, except a new atom in `blocks/`
when the operation does not exist yet.
